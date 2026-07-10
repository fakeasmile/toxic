"""仅生成18个新增形容词的概念向量（vLLM）

从v2词典中提取最后18个新增形容词，为指定数据集/模型生成概念向量。
生成后需与v1中114个有效概念向量合并，使用 merge_v2_concepts.py。

使用示例：
    python scripts/generate_v2_new_concepts.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
    python scripts/generate_v2_new_concepts.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 复用v1脚本的模型配置表和核心逻辑
from generate_adjective_c_r_vllm import (
    MODEL_LOADING_CONFIG,
    get_model_loading_config,
    load_vllm_model,
    SYSTEM_INSTRUCTION,
    build_chat_messages,
    get_first_token_ids,
    extract_likert_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="生成18个新增形容词的概念向量")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    return parser.parse_args()


def load_new_adjectives(v2_path):
    """从v2词典中提取最后18个新增形容词。"""
    with open(v2_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # v2词典结构：前114个来自v1，后18个为新增
    new_rows = rows[114:]
    if len(new_rows) != 18:
        raise ValueError(f"v2词典中新增形容词数不为18，实际为{len(new_rows)}")
    adjectives = [row[1] for row in new_rows]
    adj_english = [row[0] for row in new_rows]
    definitions = [row[2] if len(row) > 2 else None for row in new_rows]
    print(f"加载{len(new_rows)}个新增形容词:")
    for ch, en in zip(adjectives, adj_english):
        print(f"  {ch} ({en})")
    return adjectives, definitions


def generate_new_concepts(data_path, output_path, adjectives, definitions,
                          tokenizer, llm_model, is_qwen3=False, prompt_suffix="",
                          threshold=1e-4):
    """为18个新增形容词生成概念向量。"""
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)
    num_adjs = len(adjectives)

    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample in tqdm(data_set, desc="Processing samples"):
        content = sample["content"]
        prompts = []
        for adj, adj_def in zip(adjectives, definitions):
            messages = build_chat_messages(content, adj, adj_def)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_likert_score(first_token_logprobs, likert_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector长度异常：期望{num_adjs}，实际{len(concept_vector)}")

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "likert_probs": raw_probs,
        }
        results.append(result_item)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"新增概念向量保存到: {output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {num_adjs}]")

    # Verbalizer覆盖率分析
    total_scores = len(concept_matrix) * num_adjs
    nonzero = sum(1 for row in concept_matrix for s in row if s > 0)
    coverage = nonzero / total_scores
    print(f"Verbalizer覆盖率: {coverage:.2%} ({nonzero}/{total_scores})")


def main():
    args = parse_args()
    config = MLPConfig()

    v2_path = config.raw_data_path / "adjective" / "toxic_adjectives_v2.csv"
    if not v2_path.exists():
        raise FileNotFoundError(f"v2词典不存在: {v2_path}，请先运行 scripts/create_v2_dictionary.py")

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    adjectives, definitions = load_new_adjectives(v2_path)

    output_dir = config.processed_path / args.dataset_name / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"concept_{args.mode}_{args.model_name}_v2_new18.json"

    print(f"\n{'=' * 60}")
    print("生成18个新增形容词概念向量(vLLM)")
    print(f"{'=' * 60}")
    print(f"数据集: {args.dataset_name}, 模式: {args.mode}")
    print(f"模型: {args.model_name}")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print(f"{'=' * 60}\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")

    generate_new_concepts(
        data_path, output_path, adjectives, definitions,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("\n生成完成。下一步：运行 merge_v2_concepts.py 合并114+18概念向量")


if __name__ == "__main__":
    main()
