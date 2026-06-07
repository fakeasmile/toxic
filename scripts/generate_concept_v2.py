"""两阶段概念向量生成（vLLM版本）

改进点：
1. 方向2 - 改进Likert提示词：
   - 在提示词中加入形容词定义，让LLM理解每个概念的精确含义
   - 区分"表面相关"和"实质相关"，要求LLM判断文本是否真正体现了该概念
   - 对隐式毒性概念，特别强调需要推断隐含意图

2. 方向3 - 两阶段概念生成：
   - 阶段一：判断文本是否含有隐含攻击意图（是/否/不确定）
   - 阶段二：对阶段一判断为"是"或"不确定"的文本，使用改进提示词进行细粒度概念评分
   - 对阶段一判断为"否"的文本，使用标准提示词评分（但隐式概念评分会被抑制）

使用方式:
    python scripts/generate_concept_v2.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --adjective_path data/raw/adjective/merged_toxicn_concepts.csv --output_model_name Qwen2.5-7B-Instruct-AWQ-v2
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 添加项目根目录
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 模型加载配置表
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct-AWQ": {"quantization": "awq", "dtype": "auto"},
    "Qwen2.5-7B-Instruct-GPTQ-Int8": {"quantization": "gptq", "dtype": "auto"},
    "Qwen2.5-3B-Instruct": {"quantization": None, "dtype": "auto"},
    "DeepSeek-R1-Distill-Qwen-7B": {"quantization": None, "dtype": "auto"},
    "Qwen3.5-9B": {"quantization": None, "dtype": "auto"},
}

# 隐式毒性概念列表（需要特殊处理的概念）
IMPLICIT_CONCEPTS = {
    "暗示歧视的", "隐含攻击的", "语境依赖的", "话中有话的", "伪善的", "暗中贬低的",
    "正话反说的", "冷暴力的", "情感操控的", "带节奏的煽动版", "似是而非的",
    "使用谐音的", "使用暗语的", "使用缩写的", "使用表情替代的", "使用反串的",
}

# 话题-毒性区分概念（需要特别区分"讨论"vs"攻击"）
TOPIC_CONCEPTS = {
    "讨论敏感话题的", "表达不满的", "立场鲜明的", "情绪化表达的",
}


def parse_args():
    parser = argparse.ArgumentParser(description="两阶段概念向量生成（改进版）")

    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-AWQ')
    parser.add_argument('--adjective_path', type=str, default=None)
    parser.add_argument('--output_model_name', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--skip_stage1', action='store_true',
                        help='跳过阶段一，直接使用改进提示词对所有文本评分')

    return parser.parse_args()


def get_first_token_ids(word_list, tokenizer):
    """获取词语的首token id"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")
    return list(dict.fromkeys(token_ids))


def load_vllm_model(models_path, model_name, gpu_memory_utilization):
    """加载vLLM模型"""
    model_path = str(models_path / model_name)
    config = MODEL_LOADING_CONFIG.get(model_name, {"quantization": None, "dtype": "auto"})

    llm_kwargs = {
        "model": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": config["dtype"],
    }
    if config["quantization"]:
        llm_kwargs["quantization"] = config["quantization"]

    # Qwen3+ 检测
    qwen3_flag = "Qwen3" in model_name or "Qwen-3" in model_name
    if qwen3_flag:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    # 多模态模型检测
    multimodal_keywords = ["VL", "vl", "Visual", "visual", "InternVL", "MiniCPM-V"]
    if any(kw in model_name for kw in multimodal_keywords):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    llm_model = LLM(**llm_kwargs)
    tokenizer = llm_model.get_tokenizer()

    return tokenizer, llm_model, qwen3_flag


# ========== 阶段一：隐含攻击意图判断 ==========

STAGE1_SYSTEM = (
    "你是一位中文有害言论分析专家，擅长识别隐含攻击意图。\n"
    "请判断以下文本是否含有隐含的攻击、歧视或贬低意图。\n"
    "注意区分：\n"
    "- 讨论敏感话题但无攻击意图 → 否\n"
    "- 表达不满但非人身攻击 → 否\n"
    "- 使用反讽、暗示、谐音等方式隐晦攻击 → 是\n"
    "- 表面中性但实际含有歧视或贬低 → 是\n"
    "- 直接辱骂或攻击 → 是\n"
    "直接回答：是、否、不确定"
)

STAGE1_USER = "文本内容：{content}\n回答："

STAGE1_TOKENS = ["是", "否", "不"]


def run_stage1(data_set, tokenizer, llm_model, is_qwen3=False):
    """阶段一：判断每条文本是否含有隐含攻击意图

    Returns:
        list[int]: 0=否, 1=是, 2=不确定
    """
    likert_ids = get_first_token_ids(STAGE1_TOKENS, tokenizer)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 批量构建prompts
    prompts = []
    for sample in data_set:
        user_content = STAGE1_USER.format(content=sample["content"])
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompts.append(prompt_text)

    # 分批推理（避免OOM）
    batch_size = 256
    all_labels = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Stage 1: 隐含意图判断"):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm_model.generate(batch_prompts, sampling_params, use_tqdm=False)

        for output in outputs:
            logprobs = output.outputs[0].logprobs[0]
            probs_dict = {}
            for tid, logprob_obj in logprobs.items():
                probs_dict[tid] = math.exp(logprob_obj.logprob)

            # 提取 是/否/不 的概率
            yes_prob = 0.0
            no_prob = 0.0
            uncertain_prob = 0.0

            for idx, tid in enumerate(likert_ids):
                p = probs_dict.get(tid, 0.0)
                if idx == 0:  # 是
                    yes_prob = p
                elif idx == 1:  # 否
                    no_prob = p
                elif idx == 2:  # 不 (不确定的开头)
                    uncertain_prob = p

            if yes_prob > no_prob and yes_prob > uncertain_prob:
                all_labels.append(1)  # 是
            elif uncertain_prob > no_prob:
                all_labels.append(2)  # 不确定
            else:
                all_labels.append(0)  # 否

    return all_labels


# ========== 阶段二：改进的概念评分 ==========

def build_stage2_instruction(adj_name, adj_definition, is_implicit=False, is_topic=False):
    """构建改进的阶段二提示词

    根据概念类型使用不同的指令：
    - 普通概念：标准Likert + 定义
    - 隐式概念：强调需要推断隐含意图
    - 话题概念：强调区分"讨论"vs"攻击"
    """
    base = "你是一位语言分析专家。请评估以下文本在多大程度上体现了给定概念。\n"

    if is_implicit:
        base += (
            "重要：该概念涉及隐含意图，需要你推断文本表面之下的真实含义，"
            "而非仅看字面意思。如果文本只是字面上提到了相关话题但实际没有"
            "该概念描述的隐含意图，应评为低分。\n"
        )
    elif is_topic:
        base += (
            "重要：该概念描述的是一种表达方式或话题参与，而非攻击行为。"
            "讨论敏感话题、表达不满、立场鲜明或情绪化表达本身不等于有害言论。"
            "请仅评估文本是否体现了该概念描述的行为，不要因为话题敏感就给高分。\n"
        )

    base += (
        f"\n概念：{adj_name}\n"
        f"定义：{adj_definition}\n\n"
        "评估等级：\n"
        "1 = 完全不体现\n"
        "2 = 不太体现\n"
        "3 = 有点体现\n"
        "4 = 比较体现\n"
        "5 = 非常体现\n"
        "直接回答数字。"
    )

    return base


def build_stage2_messages(instruction, content):
    """构建阶段二Chat Template"""
    user_content = f"文本内容：{content}\n回答： "
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_concept_v2(data_path, output_path, csv_output_path, adjective_path,
                        temperature, tokenizer, llm_model, is_qwen3=False,
                        threshold=1e-4, stage1_labels=None):
    """两阶段概念向量生成

    Args:
        stage1_labels: 阶段一的隐含意图判断结果。None则跳过阶段一逻辑。
    """
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    # 加载形容词词典（含定义）
    adj_df = pd.read_csv(adjective_path)
    if "chinese" in adj_df.columns:
        adj_names = adj_df["chinese"].tolist()
        adj_defs = adj_df["definition"].tolist()
    elif "name" in adj_df.columns:
        adj_names = adj_df["name"].tolist()
        adj_defs = adj_df["definition"].tolist()
    else:
        raise ValueError(f"形容词词典 {adjective_path} 缺少 'chinese' 或 'name' 列")

    # 预构建每条形容词的指令
    adj_instructions = []
    for name, defn in zip(adj_names, adj_defs):
        is_implicit = name in IMPLICIT_CONCEPTS
        is_topic = name in TOPIC_CONCEPTS
        instruction = build_stage2_instruction(name, defn, is_implicit, is_topic)
        adj_instructions.append(instruction)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Stage 2: 概念评分")):
        content = sample["content"]
        has_implicit_intent = stage1_labels[sample_idx] if stage1_labels else 1  # 默认当作有

        concept_vector = []

        # 构建所有形容词的prompts
        prompts = []
        for adj_idx, (adj_name, instruction) in enumerate(zip(adj_names, adj_instructions)):
            messages = build_stage2_messages(instruction, content)

            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompts.append(prompt_text)

        # 批量推理
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        for adj_idx, sample_info in enumerate(outputs):
            logprobs = sample_info.outputs[0].logprobs
            last_token_logprobs = logprobs[0]

            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

            # 手动应用temperature
            if temperature > 0:
                logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
                adjusted_logits = {tid: l / temperature for tid, l in logits.items()}
                max_logit = max(adjusted_logits.values())
                exp_sum = sum(math.exp(l - max_logit) for l in adjusted_logits.values())
                probs_dict = {tid: math.exp(l - max_logit) / exp_sum
                              for tid, l in adjusted_logits.items()}

            # 提取1-5等级的概率
            level_probs = []
            for tid in likert_ids:
                level_probs.append(probs_dict.get(tid, 0.0))

            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            level_probs = torch.tensor(level_probs)
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()

            # 阶段一抑制：如果阶段一判断为"否"（无隐含意图），
            # 则对隐式概念和话题概念进行抑制（分数乘以衰减因子）
            adj_name = adj_names[adj_idx]
            if has_implicit_intent == 0:  # 阶段一判断无隐含意图
                if adj_name in IMPLICIT_CONCEPTS:
                    score *= 0.3  # 强抑制：隐式概念分数大幅降低
                elif adj_name in TOPIC_CONCEPTS:
                    score *= 0.7  # 弱抑制：话题概念分数适度降低

            concept_vector.append(score)

        # 截断极小值
        truncated_vector = []
        for s in concept_vector:
            if abs(s) >= threshold:
                truncated_vector.append(s)
            else:
                truncated_vector.append(0.0)

        concept_matrix.append(truncated_vector)
        results.append({"content": content, "toxic": sample["toxic"], "concept": truncated_vector})

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"概念向量(JSON)保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adj_names)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adj_names)}]")

    # 打印阶段一统计
    if stage1_labels:
        from collections import Counter
        label_counts = Counter(stage1_labels)
        total = len(stage1_labels)
        print(f"\n阶段一统计:")
        print(f"  否（无隐含意图）: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
        print(f"  是（有隐含意图）: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
        print(f"  不确定: {label_counts.get(2, 0)} ({label_counts.get(2, 0)/total*100:.1f}%)")


def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    concept_dir = config.processed_path / args.dataset_name / (args.output_model_name or args.model_name)
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}.csv"

    adjective_path = config.adjective_path
    if args.adjective_path is not None:
        adjective_path = Path(args.adjective_path)
        if not adjective_path.exists():
            raise ValueError(f"自定义形容词词典路径不存在: {adjective_path}")

    print("\n" + "=" * 60)
    print("两阶段概念向量生成(v2) - 配置信息")
    print("=" * 60)
    print(f"数据集: {args.dataset_name}")
    print(f"LLM模型: {args.model_name}")
    print(f"形容词词典: {adjective_path}")
    print(f"输出目录: {concept_dir}")
    print(f"跳过阶段一: {args.skip_stage1}")
    print(f"Temperature: {args.temperature}")

    # 加载模型
    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式")

    # 加载数据集（阶段一需要）
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # 阶段一：隐含意图判断
    stage1_labels = None
    if not args.skip_stage1:
        print("\n>>> 阶段一：判断隐含攻击意图...")
        stage1_labels = run_stage1(data_set, tokenizer, llm_model, is_qwen3=qwen3_flag)
    else:
        print("\n>>> 跳过阶段一，直接使用改进提示词评分")

    # 阶段二：改进的概念评分
    print("\n>>> 阶段二：改进提示词概念评分...")
    generate_concept_v2(
        data_path, output_path, csv_output_path, adjective_path,
        args.temperature, tokenizer, llm_model, is_qwen3=qwen3_flag,
        threshold=1e-4, stage1_labels=stage1_labels
    )

    print("\n>>> 完成!")


if __name__ == '__main__':
    main()
