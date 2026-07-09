"""生成对抗性观点概念向量（Chat Template + vLLM）

核心思路：基于ACL论文4（双空间推理框架）的对抗性观点解耦思想，
为同一条文本同时从"有害视角"和"中立视角"两个角度提取Likert概念分数，
通过两个视角的差值/拼接来区分"讨论有毒"和"实施有毒"。

【与原始generate的区别】
1. 每个文本-形容词对生成两个分数：harmful_score 和 neutral_score
2. 使用不同的system instruction引导不同视角
3. 输出格式包含双视角信息
4. 支持小数据集验证模式（--sample_size参数）

【使用方法】
# 小样本验证（AutoDL运行）
python scripts/generate_adversarial_vllm.py --dataset_name TOXICN --model_name glm-4-9b-chat --sample_size 200 --mode train
python scripts/generate_adversarial_vllm.py --dataset_name TOXICN --model_name glm-4-9b-chat --sample_size 100 --mode test
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

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
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成对抗性观点概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='训练集或测试集')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=None,
                        help='采样数量，用于小数据集验证。不指定则使用全量')
    parser.add_argument('--adjective_name', type=str, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    return parser.parse_args()


MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "",
    },
    "Qwen2.5-14B-Instruct": {
        "quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8", "is_qwen3": True, "is_multimodal": True, "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "\n",
    },
    "deepseek-llm-7b-chat": {
        "quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "",
    },
    "Baichuan2-7B-Chat": {
        "quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "",
    },
}


def get_model_loading_config(model_name: str) -> dict:
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(f"不支持的模型: {model_name}")
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")
    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path), trust_remote_code=True, dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
        max_model_len=1024, max_num_seqs=64, max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已跳过视觉编码器")

    print(f"Loading vLLM from {llm_path}")
    llm = LLM(**llm_kwargs)
    return tokenizer, llm, model_config["is_qwen3"]


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("词表中无有效token")
    return list(dict.fromkeys(token_ids))


# 对抗性双视角System Instruction
SYSTEM_INSTRUCTION_HARMFUL = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "**请从有害/攻击性视角**解读这段文本：假设该文本确实具有攻击意图或有害内容。\n"
    "在评估时，重点关注文本可能被理解为有害的方面（如字面攻击含义、隐含威胁、歧视暗示等），\n"
    "判断文本在有害视角下体现了该形容词所描述的特征的程度。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)

SYSTEM_INSTRUCTION_NEUTRAL = (
    "你是一位语言分析专家，擅长从客观中立的角度分析文本的语义特征。\n"
    "**请从中立/客观视角**解读这段文本：不考虑任何预设立场，仅根据文本的字面含义和语境，\n"
    "客观评估文本在多大程度上体现了该形容词所描述的特征。\n"
    "既不偏向有害解读，也不偏向无害解读，力求给出最中立、准确的评估。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)


def build_chat_messages(content, adj, adj_definition=None, perspective="harmful"):
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)

    if perspective == "harmful":
        sys_instruction = SYSTEM_INSTRUCTION_HARMFUL
    elif perspective == "neutral":
        sys_instruction = SYSTEM_INSTRUCTION_NEUTRAL
    else:
        raise ValueError(f"不支持的视角: {perspective}")

    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def extract_likert_score(first_token_logprobs, likert_ids):
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]
    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8
    score = (weights * level_probs_t / total).sum().item()
    return score, level_probs


def generate_adj_concept(data_path, output_path, adjective_path,
                         tokenizer, llm_model, is_qwen3=False,
                         prompt_suffix="", threshold=1e-4, sample_size=None):
    """生成对抗性双视角概念向量"""
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # 采样
    if sample_size and sample_size < len(data_set):
        np.random.seed(42)
        indices = np.random.choice(len(data_set), size=sample_size, replace=False)
        data_set = [data_set[i] for i in sorted(indices)]
        print(f"已采样 {sample_size}/{len(data_set)} 条样本")

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    results = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        harmful_scores = []
        neutral_scores = []
        harmful_probs_list = []
        neutral_probs_list = []

        prompts_harmful = []
        prompts_neutral = []

        for adj, adj_def in zip(adjectives, adj_definitions):
            msg_harmful = build_chat_messages(content, adj, adj_def, "harmful")
            msg_neutral = build_chat_messages(content, adj, adj_def, "neutral")

            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}

            p_harmful = tokenizer.apply_chat_template(
                msg_harmful, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            ) + prompt_suffix
            p_neutral = tokenizer.apply_chat_template(
                msg_neutral, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            ) + prompt_suffix

            prompts_harmful.append(p_harmful)
            prompts_neutral.append(p_neutral)

        # 批量推理有害视角
        outputs_harmful = llm_model.generate(prompts_harmful, sampling_params, use_tqdm=False)
        # 批量推理中立视角
        outputs_neutral = llm_model.generate(prompts_neutral, sampling_params, use_tqdm=False)

        for i in range(num_adjs):
            logprobs_h = outputs_harmful[i].outputs[0].logprobs[0]
            logprobs_n = outputs_neutral[i].outputs[0].logprobs[0]

            h_score, h_probs = extract_likert_score(logprobs_h, likert_ids)
            n_score, n_probs = extract_likert_score(logprobs_n, likert_ids)

            harmful_scores.append(h_score)
            neutral_scores.append(n_score)
            harmful_probs_list.append(h_probs)
            neutral_probs_list.append(n_probs)

        if len(harmful_scores) != num_adjs:
            raise RuntimeError(f"harmful_scores长度异常")

        # 计算差值向量（有害视角 - 中立视角）
        diff_vector = [h - n for h, n in zip(harmful_scores, neutral_scores)]
        truncated_diff = [s if abs(s) >= threshold else 0.0 for s in diff_vector]

        results.append({
            "content": content,
            "toxic": sample["toxic"],
            "concept_harmful": harmful_scores,
            "concept_neutral": neutral_scores,
            "concept_diff": truncated_diff,
            "likert_probs_harmful": harmful_probs_list,
            "likert_probs_neutral": neutral_probs_list,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"对抗性概念向量保存到: {output_path}")
    print(f"样本数: {len(results)}, 形容词数: {num_adjs}")


def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    adjective_path = config.adjective_path
    if args.adjective_name is not None:
        adjective_path = config.raw_data_path / "adjective" / args.adjective_name

    adj_stem = adjective_path.stem.replace("toxic_adjectives_", "")
    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_stem}_adversarial.json"

    print("\n" + "=" * 60)
    print("对抗性观点概念向量生成")
    print("=" * 60)
    print(f"数据集: {args.dataset_name}, 模式: {args.mode}, 样本数: {args.sample_size or '全量'}")
    print(f"模型: {args.model_name}")
    print(f"输出: {output_path}")
    print("=" * 60)

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")

    generate_adj_concept(
        data_path, output_path, adjective_path,
        tokenizer, llm_model, is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix, sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
