import argparse
import math
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.lrd_cbm_config import LRDCBMConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LRD-CBM: 生成rationale和概念评分")
    parser.add_argument("--dataset_name", default="TOXICN")
    parser.add_argument("--model_name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "awq", "fp8"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--skip_rationale", action="store_true", default=False)
    parser.add_argument("--skip_concepts", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def load_concepts(concept_path):
    df = pd.read_csv(concept_path)
    if "chinese" in df.columns:
        return df["chinese"].tolist()
    return df.iloc[:, 0].tolist()


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    return list(dict.fromkeys(token_ids))


def build_rationale_prompt(text):
    return f"你是一个中文有害言论分析专家。请分析以下文本是否包含有害言论，并给出详细解释。\n\n文本：{text}\n\n请按以下格式回答：\n分类：[有毒/无毒]\n解释：[详细说明该文本是否有害，包括：1.是否使用编码语言或谐音词 2.是否包含讽刺或隐含攻击 3.攻击的目标群体 4.具体的毒性表现]"


def parse_rationale(response):
    label = 0
    rationale = "无法解析"

    label_match = re.search(r"分类[：:]\s*(有毒|无毒)", response)
    if label_match:
        label_text = label_match.group(1)
        label = 1 if label_text == "有毒" else 0

    rationale_match = re.search(r"解释[：:]\s*(.+)", response, re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    return {"label": label, "rationale": rationale}


def load_vllm_model(model_path, model_name, gpu_memory_utilization=0.85, quantization=None):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {llm_path}")
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )
    return tokenizer, llm


def generate_rationales(dataset, tokenizer, llm, args, output_path):
    print("阶段1: 批量生成rationale...")
    sampling_params = SamplingParams(temperature=0.1, max_tokens=args.max_tokens)

    rationale_prompts = []
    for sample in dataset:
        prompt = build_rationale_prompt(sample["content"])
        messages = [
            {"role": "system", "content": "你是一个中文有害言论分析专家，请严格按照指定格式输出。"},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rationale_prompts.append(prompt_text)

    rationale_results = []
    for i in tqdm(range(0, len(rationale_prompts), args.batch_size), desc="Rationale生成"):
        batch_prompts = rationale_prompts[i:i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            parsed = parse_rationale(response)
            sample_idx = i + j
            rationale_results.append({
                "content": dataset[sample_idx]["content"],
                "toxic": dataset[sample_idx]["toxic"],
                "rationale": parsed["rationale"],
                "llm_label": parsed["label"],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rationale_results, f, ensure_ascii=False, indent=2)
    print(f"Rationale结果保存到: {output_path}")


def generate_concept_scores(dataset, concepts, tokenizer, llm, args, output_path, temperature):
    print("\n阶段2: 批量生成概念评分 (Likert首token方法)...")

    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家。请评估以下文本与形容词的相关程度。\n"
                   "评估等级：\n"
                   "1 = 完全不相关\n"
                   "2 = 不太相关\n"
                   "3 = 有点相关\n"
                   "4 = 比较相关\n"
                   "5 = 非常相关\n"
                   "直接回答数字。")

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    concept_results = []
    for sample_idx, sample in enumerate(tqdm(dataset, desc="概念评分生成")):
        content = sample["content"]
        prompts = []
        for adj in concepts:
            user_content = f"文本内容：{content}\n形容词：{adj}\n回答： "
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        for sample_info in outputs:
            logprobs = sample_info.outputs[0].logprobs
            last_token_logprobs = logprobs[0]

            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

            if temperature > 0:
                logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
                adjusted_logits = {tid: l / temperature for tid, l in logits.items()}
                max_logit = max(adjusted_logits.values())
                exp_sum = sum(math.exp(l - max_logit) for l in adjusted_logits.values())
                probs_dict = {tid: math.exp(l - max_logit) / exp_sum for tid, l in adjusted_logits.items()}

            level_probs = []
            for tid in likert_ids:
                level_probs.append(probs_dict.get(tid, 0.0))

            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            level_probs = torch.tensor(level_probs)
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()
            concept_vector.append(score)

        if len(concept_vector) != len(concepts):
            print(f"  警告: 样本{sample_idx}概念向量长度异常: {len(concept_vector)} vs {len(concepts)}")
            concept_vector = [0.5] * len(concepts)

        concept_results.append({
            "content": content,
            "concept_scores": concept_vector,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concept_results, f, ensure_ascii=False, indent=2)
    print(f"概念评分结果保存到: {output_path}")


def main():
    args = parse_args()
    config = LRDCBMConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.split}.json"
    concepts = load_concepts(config.concept_path)
    output_dir = config.processed_path / args.dataset_name / args.model_name / "lrd_cbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    rationale_output_path = output_dir / f"rationale_{args.split}.json"
    concept_output_path = output_dir / f"concept_scores_{args.split}.json"

    print("\n" + "=" * 60)
    print("LRD-CBM Rationale & 概念评分生成 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"数据集划分: {args.split}")
    print(f"批量大小: {args.batch_size}")
    print(f"最大生成token数: {args.max_tokens}")
    print(f"量化方法: {args.quantization if args.quantization else '无量化'}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"概念数量: {len(concepts)}")
    print(f"概念评分方法: Likert首token logprobs")
    print(f"数据集路径: {data_path}")
    print(f"Rationale输出路径: {rationale_output_path}")
    print(f"概念评分输出路径: {concept_output_path}")
    print("=" * 60 + "\n")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not args.skip_rationale:
        tokenizer, llm = load_vllm_model(
            config.base_path / "models", args.model_name,
            args.gpu_memory_utilization, args.quantization
        )
        generate_rationales(dataset, tokenizer, llm, args, rationale_output_path)
    else:
        print("跳过阶段1 (rationale已生成)")

    if not args.skip_concepts:
        if args.skip_rationale:
            tokenizer, llm = load_vllm_model(
                config.base_path / "models", args.model_name,
                args.gpu_memory_utilization, args.quantization
            )
        generate_concept_scores(dataset, concepts, tokenizer, llm, args, concept_output_path, args.temperature)
    else:
        print("跳过阶段2 (概念评分已生成)")

    print("\n生成完成!")


if __name__ == "__main__":
    main()
