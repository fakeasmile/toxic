"""生成混合概念向量（Chat Template + vLLM）

核心改进：不同类型概念使用不同提示词模板，突破单一"实施"视角的信息上限。

概念类型与提示词设计：
  1. intent（意图概念）：判断说话者的意图是否为X，二元判断（1=否 2=是）
     - 解决问题：当前"实施"锚点对隐晦毒性无效（如"某些人天生低劣"→低P(3)但高恶意意图）
     - 修正信号：当形容词概念高激活但意图概念为"中立/善意"时，应判无毒

  2. mode（表达方式概念）：判断文本是否以X方式表达，二元判断（1=否 2=是）
     - 解决问题：当前无法区分"偷图"（网络用语）和"偷窃"（字面含义）
     - 修正信号：当"网络用语"高激活时，形容词概念的"实施"可能被误导

  3. target（对象概念）：判断文本针对的对象，二元判断（1=否 2=是）
     - 解决问题：讨论毒性事件（针对事物）vs 实施毒性攻击（针对人/群体）
     - 修正信号：当"针对事物"高激活时，即使形容词概念高激活也可能无毒

  4. adjective（保留形容词）：沿用3级行为锚定提示词（1=不相关 2=涉及 3=实施）
     - 保留对显式毒性的强识别能力

使用示例：
    # 生成训练集
    python scripts/generate_mixed_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat

    # 生成测试集
    python scripts/generate_mixed_concept_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat

    # 使用自定义概念词典
    python scripts/generate_mixed_concept_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat --concept_csv mixed_concepts_v3.csv
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


# =============================================================================
# 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="生成混合概念向量（类型感知提示词 + vLLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--concept_csv', type=str, default='mixed_concepts_v3.csv',
                        help='混合概念词典文件名（位于data/raw/adjective/下）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='仅处理前N个样本（用于快速验证，默认处理全部）')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表（与generate_3level_concept_vllm.py一致）
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen2.5-14B-Instruct": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen3.5-9B": {"quantization": "fp8", "is_qwen3": True, "is_multimodal": True, "prompt_suffix": ""},
    "glm-4-9b-chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "\n"},
    "deepseek-llm-7b-chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Baichuan2-7B-Chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen3-8B": {"quantization": None, "is_qwen3": True, "is_multimodal": False, "prompt_suffix": ""},
}


def get_model_loading_config(model_name: str) -> dict:
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。")
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 类型感知提示词定义
# =============================================================================

# --- 意图概念提示词 ---
INTENT_SYSTEM = (
    "你是一位语言分析专家，擅长识别文本中说话者的真实意图。\n"
    "你的任务是判断文本中说话者的意图是否为指定概念所描述的意图。\n"
    "请从以下两个选项中选择：\n"
    "1 — 否：说话者的意图不是该概念描述的意图\n"
    "2 — 是：说话者的意图是该概念描述的意图\n"
    "判断要点：\n"
    "- 关注说话者的真实意图，而非字面含义\n"
    "- 区分"表达某意图"与"提及/讨论某意图"：说话者自身在表达该意图选"是"，仅客观讨论选"否"\n"
    "- 对于隐含意图（如暗示、反讽），需透过字面理解真实意图\n"
    "只回答一个数字（1或2）。"
)

# --- 表达方式概念提示词 ---
MODE_SYSTEM = (
    "你是一位语言分析专家，擅长识别文本的表达方式。\n"
    "你的任务是判断文本是否以指定概念描述的方式表达。\n"
    "请从以下两个选项中选择：\n"
    "1 — 否：文本不是以该方式表达的\n"
    "2 — 是：文本是以该方式表达的\n"
    "判断要点：\n"
    "- 关注文本的实际表达方式，而非内容本身\n"
    "- 一种文本可能同时具有多种表达方式，请独立判断每种方式\n"
    "- 网络用语包括：谐音梗、缩写、网络流行语、梗文化表达等\n"
    "只回答一个数字（1或2）。"
)

# --- 对象概念提示词 ---
TARGET_SYSTEM = (
    "你是一位语言分析专家，擅长识别文本的指向对象。\n"
    "你的任务是判断文本是否针对指定概念描述的对象。\n"
    "请从以下两个选项中选择：\n"
    "1 — 否：文本不是针对该对象的\n"
    "2 — 是：文本是针对该对象的\n"
    "判断要点：\n"
    "- 关注文本的核心指向，即说话者主要在评价、讨论或作用于谁\n"
    "- 一种文本可能同时针对多种对象，请独立判断每种对象\n"
    "- "针对个人/群体"意味着文本在评价、批评或攻击该对象\n"
    "- "针对事物"意味着文本在讨论事件、现象或抽象概念\n"
    "只回答一个数字（1或2）。"
)

# --- 形容词概念提示词（沿用3级） ---
ADJ_SYSTEM = (
    "你是一位语言分析专家，擅长识别文本中的言语行为意图。\n"
    "你的任务是判断文本中说话者与指定概念之间的关系，从三个级别中选择最匹配的：\n"
    "1 — 不相关：文本与该概念描述的内容无关\n"
    "2 — 涉及：文本涉及该概念，但说话者是在提及、讨论、分析、引用或反驳，而非自身实施\n"
    "3 — 实施：说话者本人正在实施该概念描述的言行\n"
    "\"涉及\"与\"实施\"的关键区分：\n"
    "- 涉及：说话者在谈论该行为，但自身并非行为的执行者\n"
    "  例如：分析辱骂现象、报告歧视事件、引用他人攻击性言论、反驳偏见观点\n"
    "- 实施：说话者本人就是该行为的执行者\n"
    "  例如：说话者本人正在辱骂、威胁、歧视、煽动，或以比喻、反讽、谐音等隐含方式实施\n"
    "请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等）。\n"
    "只回答一个数字（1、2或3）。"
)


# =============================================================================
# 提示词构建（类型感知）
# =============================================================================
def build_messages(content, concept_name, concept_def, concept_type):
    """根据概念类型构建不同的Chat Template messages。"""

    if concept_type == "intent":
        system_msg = INTENT_SYSTEM
        user_lines = [f"文本内容：{content}"]
        user_lines.append(f"意图：{concept_name}")
        user_lines.append(f"定义：{concept_def}")
        user_lines.append("说话者的意图是否为上述意图？（1=否 2=是）回答：")
        user_content = "\n".join(user_lines)

    elif concept_type == "mode":
        system_msg = MODE_SYSTEM
        user_lines = [f"文本内容：{content}"]
        user_lines.append(f"表达方式：{concept_name}")
        user_lines.append(f"定义：{concept_def}")
        user_lines.append("文本是否以上述方式表达？（1=否 2=是）回答：")
        user_content = "\n".join(user_lines)

    elif concept_type == "target":
        system_msg = TARGET_SYSTEM
        user_lines = [f"文本内容：{content}"]
        user_lines.append(f"指向对象：{concept_name}")
        user_lines.append(f"定义：{concept_def}")
        user_lines.append("文本是否针对上述对象？（1=否 2=是）回答：")
        user_content = "\n".join(user_lines)

    elif concept_type == "adjective":
        system_msg = ADJ_SYSTEM
        user_lines = [f"文本内容：{content}"]
        user_lines.append(f"概念：{concept_name}")
        user_lines.append(f"定义：{concept_def}")
        user_lines.append(f"说话者与\"{concept_name}\"的关系级别（1=不相关 2=涉及 3=实施）：回答：")
        user_content = "\n".join(user_lines)

    else:
        raise ValueError(f"未知概念类型: {concept_type}")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))


def extract_score(first_token_logprobs, level_ids):
    """从首token的logprobs中提取概率分布。

    Returns:
        (score, level_probs): 
            score: 主信号分数（二元=P(2), 3级=P(3)）
            level_probs: 完整概率列表
    """
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in level_ids]
    total = sum(level_probs) + 1e-8
    level_probs = [p / total for p in level_probs]  # 归一化

    score = level_probs[-1]  # 二元→P(2)=P(是), 3级→P(3)=P(实施)

    return score, level_probs


# =============================================================================
# 核心流程
# =============================================================================
def generate_mixed_concept(data_path, output_path, concept_csv_path,
                           tokenizer, llm_model,
                           is_qwen3=False, prompt_suffix="", threshold=1e-4,
                           num_samples=None):
    """生成混合概念向量。

    不同概念类型使用不同的提示词模板和verbalizer：
    - intent/mode/target: 二元verbalizer {"1", "2"}
    - adjective: 三级verbalizer {"1", "2", "3"}
    """
    # 加载概念词典
    concept_df = pd.read_csv(concept_csv_path)
    concept_names = concept_df["name"].tolist()
    concept_types = concept_df["type"].tolist()
    concept_defs = concept_df["definition"].tolist() if "definition" in concept_df.columns else [None] * len(concept_names)
    num_concepts = len(concept_names)

    # 统计各类型数量
    type_counts = {}
    for t in concept_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"概念类型分布: {type_counts}")

    # Verbalizer: 二元和三级
    binary_ids = get_first_token_ids(["1", "2"], tokenizer)
    three_level_ids = get_first_token_ids(["1", "2", "3"], tokenizer)

    print(f"二元Verbalizer token IDs: {binary_ids}")
    print(f"三级Verbalizer token IDs: {three_level_ids}")

    # 为每个概念确定verbalizer
    concept_level_ids = []
    for t in concept_types:
        if t in ("intent", "mode", "target"):
            concept_level_ids.append(binary_ids)
        else:
            concept_level_ids.append(three_level_ids)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    if num_samples is not None:
        data_set = data_set[:num_samples]
        print(f"限制处理前 {num_samples} 个样本")

    # vLLM推理
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample in tqdm(data_set, desc="Processing samples"):
        content = sample["content"]

        prompts = []
        for i in range(num_concepts):
            messages = build_messages(content, concept_names[i], concept_defs[i], concept_types[i])
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []
        concept_levels = []  # 记录每个概念的级别数

        for i, sample_info in enumerate(outputs):
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_score(first_token_logprobs, concept_level_ids[i])
            concept_vector.append(score)
            raw_probs.append(level_probs)
            concept_levels.append(len(level_probs))

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "level_probs": raw_probs,
            "concept_names": concept_names,
            "concept_types": concept_types,
            "concept_levels": concept_levels,
        }
        results.append(result_item)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"混合概念向量保存到: {output_path}")

    # 统计
    total_scores = len(concept_matrix) * num_concepts
    nonzero = sum(1 for row in concept_matrix for s in row if s > 0)
    coverage = nonzero / total_scores

    print(f"概念矩阵形状: [{len(concept_matrix)}, {num_concepts}]")
    print(f"概念激活率: {coverage:.2%} ({nonzero}/{total_scores})")

    # 按类型统计概率分布
    for ctype in ["intent", "mode", "target", "adjective"]:
        type_indices = [i for i, t in enumerate(concept_types) if t == ctype]
        if not type_indices:
            continue
        n_levels = 2 if ctype in ("intent", "mode", "target") else 3
        level_means = [0.0] * n_levels
        count = 0
        for r in results:
            for idx in type_indices:
                probs = r["level_probs"][idx]
                for lv in range(n_levels):
                    level_means[lv] += probs[lv]
                count += 1
        if count > 0:
            level_means = [m / count for m in level_means]
            if n_levels == 2:
                print(f"  {ctype}: P(1=否)={level_means[0]:.4f}, P(2=是)={level_means[1]:.4f}")
            else:
                print(f"  {ctype}: P(1)={level_means[0]:.4f}, P(2)={level_means[1]:.4f}, P(3)={level_means[2]:.4f}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    concept_csv_path = config.raw_data_path / "adjective" / args.concept_csv

    if not concept_csv_path.exists():
        raise FileNotFoundError(f"概念词典不存在: {concept_csv_path}")

    concept_stem = Path(args.concept_csv).stem  # e.g., mixed_concepts_v3

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{concept_stem}_mixed.json"

    print("\n" + "=" * 60)
    print("混合概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"概念词典: {concept_csv_path.name}")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    if args.num_samples:
        print(f"样本限制: 前{args.num_samples}条")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式")

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    generate_mixed_concept(
        data_path, output_path, concept_csv_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
        num_samples=args.num_samples,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
