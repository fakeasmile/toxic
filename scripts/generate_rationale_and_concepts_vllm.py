"""LRD-CBM方案离线LLM推理脚本

使用vLLM框架批量生成rationale和概念评分。

使用示例：
python scripts/generate_rationale_and_concepts_vllm.py --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --split train
python scripts/generate_rationale_and_concepts_vllm.py --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --split test --quantization awq
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
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
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return parser.parse_args()


def load_concepts(concept_path):
    df = pd.read_csv(concept_path)
    if "chinese" in df.columns:
        return df["chinese"].tolist()
    return df.iloc[:, 0].tolist()


def build_rationale_prompt(text):
    return f"你是一个中文有害言论分析专家。请分析以下文本是否包含有害言论，并给出详细解释。\n\n文本：{text}\n\n请按以下格式回答：\n分类：[有毒/无毒]\n解释：[详细说明该文本是否有害，包括：1.是否使用编码语言或谐音词 2.是否包含讽刺或隐含攻击 3.攻击的目标群体 4.具体的毒性表现]"


def build_concept_prompt(text, concepts):
    concepts_str = "、".join(concepts)
    return f"请对以下文本在各个概念维度上进行1-5分评分（1=完全不相关，5=高度相关）。\n\n文本：{text}\n概念列表：{concepts_str}\n\n请按JSON格式输出评分，例如：{{\"攻击性\": 4, \"侮辱性\": 3, ...}}"


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


def parse_concept_scores(response, concepts):
    scores = [0.5] * len(concepts)

    json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            for i, concept in enumerate(concepts):
                if concept in parsed:
                    val = parsed[concept]
                    if isinstance(val, (int, float)):
                        scores[i] = max(0.0, min(1.0, (val - 1) / 4))
            return scores
        except json.JSONDecodeError:
            pass

    number_pattern = re.findall(r"\"[^\"]+\"[：:]\s*(\d)", response)
    if number_pattern and len(number_pattern) >= len(concepts):
        for i in range(len(concepts)):
            val = int(number_pattern[i])
            scores[i] = max(0.0, min(1.0, (val - 1) / 4))

    return scores


def load_vllm_model(model_path, model_name, gpu_memory_utilization=0.9, quantization=None):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
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
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
    )
    return tokenizer, llm


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
    print(f"数据集路径: {data_path}")
    print(f"Rationale输出路径: {rationale_output_path}")
    print(f"概念评分输出路径: {concept_output_path}")
    print("=" * 60 + "\n")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenizer, llm = load_vllm_model(
        config.base_path / "models", args.model_name,
        args.gpu_memory_utilization, args.quantization
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=args.max_tokens)

    print("阶段1: 批量生成rationale...")
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

    with open(rationale_output_path, "w", encoding="utf-8") as f:
        json.dump(rationale_results, f, ensure_ascii=False, indent=2)
    print(f"Rationale结果保存到: {rationale_output_path}")

    print("\n阶段2: 批量生成概念评分...")
    concept_prompts = []
    for sample in dataset:
        prompt = build_concept_prompt(sample["content"], concepts)
        messages = [
            {"role": "system", "content": "你是一个语言分析专家，请严格按照JSON格式输出评分。"},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        concept_prompts.append(prompt_text)

    concept_results = []
    for i in tqdm(range(0, len(concept_prompts), args.batch_size), desc="概念评分生成"):
        batch_prompts = concept_prompts[i:i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            scores = parse_concept_scores(response, concepts)
            sample_idx = i + j
            concept_results.append({
                "content": dataset[sample_idx]["content"],
                "concept_scores": scores,
            })

    with open(concept_output_path, "w", encoding="utf-8") as f:
        json.dump(concept_results, f, ensure_ascii=False, indent=2)
    print(f"概念评分结果保存到: {concept_output_path}")

    print(f"\n生成完成! Rationale: {len(rationale_results)}条, 概念评分: {len(concept_results)}条")


if __name__ == "__main__":
    main()
