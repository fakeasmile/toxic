"""生成语义+语用双分量概念向量（Chat Template + vLLM）

核心思路：每个形容词获取两个正交维度的分数：
1. 语义分量（Likert 1-5）：文本内容与形容词的语义相关程度
2. 语用分量（P(USE)）：说话者是在主动表达(USE)还是仅提及(MENTION)

【信息论基础】
不同视角的Likert分数是对LLM同一内部表示的不同投影，互信息上限受限。
而Likert(语义相关性) vs P(USE)(语用功能)改变的是LLM计算的目标函数：
- Likert问"文本内容与该特征有多相关？" → 有毒文本和讨论有毒的文本都得高分
- USE/MENTION问"说话者是在表达还是提及？" → 有毒文本P(USE)高，讨论有毒P(USE)低

【组合向量】
177维Likert + 177维P(USE) = 354维，两个分量信息来源不同。

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
        description="生成语义+语用双分量概念向量（vLLM版本）",
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


# ==================== 语义分量：Likert 1-5 ====================
SYSTEM_INSTRUCTION_LIKERT = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "**请从有害/攻击性视角**解读这段文本：假设该文本确实具有攻击意图或有害内容。\n"
    "在评估时，重点关注文本可能被理解为有害的方面（如字面攻击含义、隐含威胁、歧视暗示等），\n"
    "判断文本在有害视角下体现了该形容词所描述的特征的程度。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)


# ==================== 语用分量：USE vs MENTION ====================
SYSTEM_INSTRUCTION_USE_MENTION = (
    "你是一位语用学专家，擅长区分语言的使用（USE）和提及（MENTION）。\n"
    "当说话者主动表达、认同或主张某个特征时，属于USE（使用）；\n"
    "当说话者仅仅在引用、转述、质疑、反问或讨论中提及该特征时，属于MENTION（提及）。\n"
    "判断该文本中与指定形容词相关的内容，说话者是在USE（主动表达/认同）还是MENTION（引用/提及）。"
)


def build_likert_messages(content, adj, adj_definition=None):
    """构建Likert语义评估prompt"""
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION_LIKERT},
        {"role": "user", "content": user_content},
    ]


def build_use_mention_messages(content, adj, adj_definition=None):
    """构建USE/MENTION语用评估prompt"""
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(
        f"文本中与\"{adj}\"相关的内容，说话者是在主动表达/认同（USE），"
        f"还是仅在引用/提及（MENTION）？\n"
        f"回答A表示USE（主动表达/认同），回答B表示MENTION（引用/提及）。只回答A或B。"
    )
    user_content = "\n".join(user_lines)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION_USE_MENTION},
        {"role": "user", "content": user_content},
    ]


def extract_likert_score(first_token_logprobs, likert_ids):
    """提取Likert加权期望分数（语义分量）"""
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]
    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8
    score = (weights * level_probs_t / total).sum().item()
    return score, level_probs


def extract_use_prob(first_token_logprobs, use_id, mention_id):
    """提取P(USE)概率（语用分量）"""
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    p_use = probs_dict.get(use_id, 0.0)
    p_mention = probs_dict.get(mention_id, 0.0)
    total = p_use + p_mention + 1e-8
    use_prob = p_use / total
    return use_prob, [p_use, p_mention]


def generate_dual_concept(data_path, output_path, adjective_path,
                          tokenizer, llm_model, is_qwen3=False,
                          prompt_suffix="", sample_size=None):
    """生成语义+语用双分量概念向量"""
    # Likert verbalizer tokens
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    # USE/MENTION verbalizer tokens
    use_mention_tokens = ["A", "B"]
    um_ids = get_first_token_ids(use_mention_tokens, tokenizer)
    use_id = um_ids[0]
    mention_id = um_ids[1]
    print(f"USE token ID: {use_id}, MENTION token ID: {mention_id}")

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

        prompts_likert = []
        prompts_um = []

        for adj, adj_def in zip(adjectives, adj_definitions):
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}

            msg_likert = build_likert_messages(content, adj, adj_def)
            p_likert = tokenizer.apply_chat_template(
                msg_likert, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            ) + prompt_suffix
            prompts_likert.append(p_likert)

            msg_um = build_use_mention_messages(content, adj, adj_def)
            p_um = tokenizer.apply_chat_template(
                msg_um, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            ) + prompt_suffix
            prompts_um.append(p_um)

        # 批量推理：语义分量
        outputs_likert = llm_model.generate(prompts_likert, sampling_params, use_tqdm=False)
        # 批量推理：语用分量
        outputs_um = llm_model.generate(prompts_um, sampling_params, use_tqdm=False)

        likert_scores = []
        use_probs = []
        likert_probs_list = []
        um_probs_list = []

        for i in range(num_adjs):
            # 语义分量
            logprobs_lk = outputs_likert[i].outputs[0].logprobs[0]
            lk_score, lk_probs = extract_likert_score(logprobs_lk, likert_ids)
            likert_scores.append(lk_score)
            likert_probs_list.append(lk_probs)

            # 语用分量
            logprobs_um = outputs_um[i].outputs[0].logprobs[0]
            use_prob, um_probs = extract_use_prob(logprobs_um, use_id, mention_id)
            use_probs.append(use_prob)
            um_probs_list.append(um_probs)

        if len(likert_scores) != num_adjs:
            raise RuntimeError("likert_scores长度异常")

        # 组合向量：177维Likert + 177维P(USE) = 354维
        combined = likert_scores + use_probs

        results.append({
            "content": content,
            "toxic": sample["toxic"],
            "concept_likert": likert_scores,
            "concept_use_prob": use_probs,
            "concept_combined": combined,
            "likert_probs": likert_probs_list,
            "use_mention_probs": um_probs_list,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"双分量概念向量保存到: {output_path}")
    print(f"样本数: {len(results)}, 形容词数: {num_adjs}")
    print(f"语义分量(likert): {num_adjs}维, 语用分量(use_prob): {num_adjs}维, 组合: {num_adjs*2}维")


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
    print("语义+语用双分量概念向量生成")
    print("语义分量: Likert 1-5 (有害视角)")
    print("语用分量: P(USE) 二分类 (主动表达 vs 引用提及)")
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

    generate_dual_concept(
        data_path, output_path, adjective_path,
        tokenizer, llm_model, is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix, sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
