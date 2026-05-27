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

from configs.cotd_cbm_config import CoTDCBMConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="TOXICN")
    parser.add_argument("--model_name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
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


def build_soft_label_prompt(text):
    return f"你是一个中文有害言论分析专家。请分析以下文本是否包含有害言论。\n\n文本：{text}\n\n请先思考该文本的特征，然后给出分类和置信度。\n\n思考过程："


def parse_soft_label_response(response):
    label = 0
    confidence = 0.5
    rationale = "无法解析"

    label_match = re.search(r"分类[：:]\s*(有毒|无毒)", response)
    if label_match:
        label = 1 if label_match.group(1) == "有毒" else 0

    conf_match = re.search(r"置信度[：:]\s*(\d+)", response)
    if conf_match:
        confidence = min(1.0, max(0.0, int(conf_match.group(1)) / 100.0))

    rationale_match = re.search(r"思考过程[：:]\s*(.+?)(?=分类[：:]|$)", response, re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    else:
        after_think = response.split("思考过程")[-1] if "思考过程" in response else response
        rationale = after_think.strip()

    if label == 1:
        soft_label = [1.0 - confidence, confidence]
    else:
        soft_label = [confidence, 1.0 - confidence]

    return {"label": label, "confidence": confidence, "soft_label": soft_label, "rationale": rationale}


def load_vllm_model(model_path, model_name, gpu_memory_utilization=0.85, quantization=None):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def main():
    args = parse_args()
    config = CoTDCBMConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.split}.json"
    concepts = load_concepts(config.concept_path)
    output_dir = config.processed_path / args.dataset_name / args.model_name / "cotd_cbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"distill_{args.split}.json"

    print(f"数据集: {args.dataset_name}, 划分: {args.split}")
    print(f"概念数量: {len(concepts)}")
    print(f"输出路径: {output_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenizer, llm = load_vllm_model(config.models_path, args.model_name, args.gpu_memory_utilization, args.quantization)

    print("\n阶段1: 生成软标签+rationale...")
    sampling_params = SamplingParams(temperature=0.1, max_tokens=args.max_tokens)

    soft_label_prompts = []
    for sample in dataset:
        prompt = build_soft_label_prompt(sample["content"])
        messages = [
            {"role": "system", "content": "你是一个中文有害言论分析专家，请严格按照指定格式输出。"},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        soft_label_prompts.append(prompt_text)

    results = []
    for i in tqdm(range(0, len(soft_label_prompts), args.batch_size), desc="软标签生成"):
        batch_prompts = soft_label_prompts[i:i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            parsed = parse_soft_label_response(response)
            sample_idx = i + j
            results.append({
                "content": dataset[sample_idx]["content"],
                "toxic": dataset[sample_idx]["toxic"],
                "soft_label": parsed["soft_label"],
                "llm_label": parsed["label"],
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
            })

    print("\n阶段2: 生成概念评分...")
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家。请评估以下文本与形容词的相关程度。\n"
                   "评估等级：\n1 = 完全不相关\n2 = 不太相关\n3 = 有点相关\n4 = 比较相关\n5 = 非常相关\n直接回答数字。")

    concept_sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

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

        outputs = llm.generate(prompts, concept_sampling_params, use_tqdm=False)

        concept_vector = []
        for sample_info in outputs:
            logprobs = sample_info.outputs[0].logprobs
            last_token_logprobs = logprobs[0]

            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

            if args.temperature > 0:
                logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
                adjusted_logits = {tid: l / args.temperature for tid, l in logits.items()}
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
            concept_vector = [0.5] * len(concepts)

        results[sample_idx]["concept_scores"] = concept_vector

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果保存到: {output_path}")


if __name__ == "__main__":
    main()
