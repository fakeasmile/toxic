"""基于发现的概念词汇表生成概念向量（Chat Template, vLLM版本）

与generate_adjective_c_r_vllm.py的核心逻辑一致，但使用discover_concepts_vllm.py
发现的概念词汇表替代原始的177个翻译形容词。

执行流程：
1. 加载vLLM模型和tokenizer
2. 加载概念词汇表（discovered_concepts.csv）
3. 对每条文本，为所有概念构建Likert评分提示词
4. vLLM批量推理，提取首token logprobs
5. 计算Likert加权期望得分，组成概念向量
6. 保存JSON和CSV文件

使用示例：
python scripts/generate_concept_vectors_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --quantization awq
"""

import argparse
import math
import sys
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.rcwn_config import RCWNConfig


def parse_args():
    parser = argparse.ArgumentParser(description="基于发现的概念词汇表生成概念向量")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-AWQ')
    parser.add_argument('--quantization', type=str, default='awq', choices=[None, 'awq', 'fp8'])
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--concept_vocab_path', type=str, default=None)
    return parser.parse_args()


def load_vllm_model(model_path, model_name, gpu_memory_utilization=0.85, quantization=None):
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
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )
    return tokenizer, llm


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")
    return list(dict.fromkeys(token_ids))


def build_chat_messages(instruction, content, concept_name):
    user_content = f"文本内容：{content}\n概念：{concept_name}\n回答： "
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_concept_vectors(data_path, output_path, csv_output_path, concept_vocab_path,
                             temperature, tokenizer, llm_model, threshold=1e-4):
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家。请评估以下文本与给定概念的相关程度。\n"
                   "评估等级：\n"
                   "1 = 完全不相关\n"
                   "2 = 不太相关\n"
                   "3 = 有点相关\n"
                   "4 = 比较相关\n"
                   "5 = 非常相关\n"
                   "直接回答数字。")

    concepts_df = pd.read_csv(concept_vocab_path)
    concept_names = concepts_df["name"].tolist()

    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]
        concept_vector = []

        prompts = []
        for concept_name in concept_names:
            messages = build_chat_messages(instruction, content, concept_name)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

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
                probs_dict = {tid: math.exp(l - max_logit) / exp_sum
                              for tid, l in adjusted_logits.items()}

            level_probs = []
            for tid in likert_ids:
                level_probs.append(probs_dict.get(tid, 0.0))

            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            level_probs = torch.tensor(level_probs)
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()

            concept_vector.append(score)

        if len(concept_vector) != len(concept_names):
            raise RuntimeError(
                f"concept_vector length mismatch: expected {len(concept_names)}, got {len(concept_vector)}"
            )

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        results.append({"content": content, "toxic": sample["toxic"], "concept": truncated_vector})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"概念向量(JSON)保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=concept_names)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(concept_names)}]")


def main():
    args = parse_args()
    config = RCWNConfig()

    concept_vocab_path = Path(args.concept_vocab_path) if args.concept_vocab_path else config.concept_vocab_path
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    concept_dir = (config.processed_path / args.dataset_name / args.model_name
                   / "rcwn_concepts")
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}.csv"

    print("\n" + "=" * 60)
    print("概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集: {args.dataset_name}")
    print(f"LLM模型: {args.model_name}")
    print(f"概念词汇表: {concept_vocab_path}")
    print(f"当前模式: {args.mode}")
    print(f"量化方法: {args.quantization or '无量化'}")
    print(f"采样温度: {args.temperature}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model = load_vllm_model(
        config.models_path, args.model_name,
        args.gpu_memory_utilization, args.quantization
    )
    generate_concept_vectors(
        data_path, output_path, csv_output_path, concept_vocab_path,
        args.temperature, tokenizer, llm_model
    )
    print("生成完成")


if __name__ == '__main__':
    main()
