"""LLM驱动的中文有害言论概念发现

使用vLLM加载LLM，分析TOXICN训练数据，自动发现中文特有的有害言论概念。
替代当前从英文翻译的177个形容词，生成更贴合中文语境的概念词汇表。

执行流程：
1. 从TOXICN训练集中采样有毒/无毒文本
2. 构建概念发现提示词，让LLM分析区分性概念
3. 多轮迭代发现，去重合并
4. 生成概念词汇表CSV文件

使用示例：
python scripts/discover_concepts_vllm.py --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --quantization awq
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.rcwn_config import RCWNConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LLM驱动的中文有害言论概念发现")
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-AWQ')
    parser.add_argument('--quantization', type=str, default='awq', choices=[None, 'awq', 'fp8'])
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--sample_size', type=int, default=200)
    parser.add_argument('--concepts_per_round', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output_path', type=str, default=None)
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
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
    )
    return tokenizer, llm


def sample_data(data_path, sample_size, seed=1):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    toxic = [d for d in data if d["toxic"] == 1]
    non_toxic = [d for d in data if d["toxic"] == 0]

    half = sample_size // 2
    import random
    random.seed(seed)
    toxic_sample = random.sample(toxic, min(half, len(toxic)))
    non_toxic_sample = random.sample(non_toxic, min(half, len(non_toxic)))

    return toxic_sample, non_toxic_sample


def build_discovery_prompt(toxic_texts, non_toxic_texts, concepts_per_round, existing_concepts=None):
    toxic_str = "\n".join([f"- {t['content']}" for t in toxic_texts[:20]])
    non_toxic_str = "\n".join([f"- {t['content']}" for t in non_toxic_texts[:20]])

    existing_str = ""
    if existing_concepts:
        existing_str = f"\n\n已发现的概念（请勿重复）：{', '.join(existing_concepts)}"

    prompt = f"""你是一位语言分析专家，专注于中文有害言论检测。以下是来自社交媒体的中文文本样本。

【有毒文本样本】：
{toxic_str}

【无毒文本样本】：
{non_toxic_str}
{existing_str}

请分析这些文本，列出能够区分有毒和无毒文本的关键属性概念。要求：
1. 每个概念用一个简短的中文形容词或短语命名（2-6个字）
2. 每个概念给出明确的定义（一句话）
3. 概念应覆盖以下维度：攻击性、歧视性、情绪性、表达方式、目标指向、文化特异性
4. 特别关注中文网络语境下的特有概念（如阴阳怪气、带节奏、扣帽子等）
5. 列出{concepts_per_round}个最具有区分度的概念

请严格按以下格式输出，每行一个概念：
概念名 | 定义"""
    return prompt


def parse_concepts(text):
    concepts = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            parts = line.split("|", 1)
            name = parts[0].strip().lstrip("0123456789.-) ")
            definition = parts[1].strip() if len(parts) > 1 else ""
            if name and len(name) <= 10:
                concepts.append({"name": name, "definition": definition})
        else:
            match = re.match(r'^[\d]+[.、)]\s*(.+)', line)
            if match:
                name = match.group(1).strip()
                if name and len(name) <= 10:
                    concepts.append({"name": name, "definition": ""})
    return concepts


def discover_concepts(tokenizer, llm, toxic_texts, non_toxic_texts,
                      concepts_per_round, rounds):
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    all_concepts = []
    existing_names = set()

    for round_idx in range(rounds):
        random_toxic = toxic_texts
        random_non_toxic = non_toxic_texts

        prompt = build_discovery_prompt(
            random_toxic, random_non_toxic, concepts_per_round,
            list(existing_names) if existing_names else None
        )

        messages = [
            {"role": "system", "content": "你是一位语言分析专家，请严格按照指定格式输出。"},
            {"role": "user", "content": prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{rounds} - LLM Response:")
        print(response[:500])
        print(f"{'='*60}")

        new_concepts = parse_concepts(response)
        for c in new_concepts:
            if c["name"] not in existing_names:
                all_concepts.append(c)
                existing_names.add(c["name"])

        print(f"Round {round_idx + 1}: discovered {len(new_concepts)} new, total {len(all_concepts)}")

    return all_concepts


def deduplicate_concepts(concepts):
    seen = set()
    unique = []
    for c in concepts:
        name = c["name"]
        if name not in seen:
            seen.add(name)
            unique.append(c)
    return unique


def main():
    args = parse_args()
    config = RCWNConfig()

    data_path = config.raw_data_path / args.dataset_name / "train.json"
    output_path = Path(args.output_path) if args.output_path else config.concept_vocab_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("LLM概念发现 - 配置信息")
    print("=" * 60)
    print(f"数据集: {args.dataset_name}")
    print(f"LLM模型: {args.model_name}")
    print(f"采样数量: {args.sample_size}")
    print(f"每轮概念数: {args.concepts_per_round}")
    print(f"发现轮数: {args.rounds}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm = load_vllm_model(
        config.models_path, args.model_name,
        args.gpu_memory_utilization, args.quantization
    )

    toxic_texts, non_toxic_texts = sample_data(data_path, args.sample_size, config.seed)
    print(f"采样: {len(toxic_texts)} toxic, {len(non_toxic_texts)} non-toxic")

    concepts = discover_concepts(
        tokenizer, llm, toxic_texts, non_toxic_texts,
        args.concepts_per_round, args.rounds
    )

    concepts = deduplicate_concepts(concepts)
    print(f"\n去重后概念总数: {len(concepts)}")

    df = pd.DataFrame(concepts)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"概念词汇表已保存到: {output_path}")

    print("\n概念列表:")
    for i, c in enumerate(concepts):
        print(f"  {i+1}. {c['name']} - {c['definition']}")


if __name__ == '__main__':
    main()
