"""Verbalizer覆盖率全景分析工具（全形容词扫描，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm.py 的"全形容词切片"评估工具。
generate_adjective_c_r_vllm.py 负责为数据集中所有文本、所有形容词批量生成概念向量；
inspect_prompt_template_vllm.py 负责在单样本级别（一个文本 + 一个形容词）调试提示词和 verbalizer；
而本脚本则对"一条固定文本 + 全部形容词（约236个）"进行扫描，评估该提示词模板和 verbalizer 词表
在整个形容词词典上的覆盖能力是否稳定。

【核心功能】
对单条文本遍历所有形容词，使用 vLLM 推理并提取 verbalizer token 的概率总和：
- binary / ICL 模板：分别计算 pos_prob（肯定词概率总和）、neg_prob（否定词概率总和）、
  total_prob = pos_prob + neg_prob（verbalizer 总概率）
- likert 模板：计算 total_prob = level_1 + ... + level_5（1-5 数字 verbalizer 总概率）

【评估标准】
- total_prob 理想区间：0.6 ~ 1.0（多数形容词应在此范围内）
  - 接近 1.0：模型首 token 概率质量高度集中在 verbalizer 词表内，提示词约束能力强，
    verbalizer 词表统计完整。
  - 低于 0.5：模型大量概率分散到非预期词（如"我"、"这"、"可能"等），说明提示词模板
    未能有效约束输出方向，或 verbalizer 词表遗漏了模型偏好的表达形式，需改进。
- 通过观察 total_prob 在不同形容词上的波动，可识别出哪些类型的形容词容易导致模型输出失控，
  从而针对性优化提示词或扩充 verbalizer 词表。

【与 generate_adjective_c_r_vllm.py / inspect_prompt_template_vllm.py 的关系】
- 本脚本的提示词构建逻辑、verbalizer 词表、分数计算逻辑与 
  generate_adjective_c_r_vllm.py 完全一致。
- inspect_prompt_template_vllm.py 用于"点"级别的单样本调试（快速迭代提示词和 verbalizer）；
- 本脚本用于"面"级别的全景验证（确认改进后的模板和 verbalizer 在整个形容词词典上表现稳定）；
- 两者结合，确保 generate_adjective_c_r_vllm.py 批量生成的概念向量质量可靠。

【输出】
1. 可视化图表（PNG）：横轴为形容词索引，纵轴为概率值
   - binary / ICL：三条线（pos_prob 绿色、neg_prob 红色、total_prob 蓝色）+ total 均值参考线
   - likert：一条线（total_prob 蓝色）+ 均值参考线
2. JSON 数据文件：每个形容词的详细概率数据 + 统计摘要（均值 / 最小值 / 最大值）

【使用方法】
1. 修改下方 CONFIG 区域的变量（模型名、模板类型、文本内容等）
2. 运行：python scripts/inspect_verbalizer_coverage_vllm.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "Qwen2.5-3B-Instruct"  # models目录下的模型文件夹名

# 提示词模板类型，可选: "binary", "likert", "ICL"
PROMPT_TEMPLATE = "likert"

# 文本内容（直接修改即可）
TEXT_CONTENT = "中国男性根本就配不上中国女性"

# 输出目录（相对于项目根目录）
OUTPUT_DIR = "experiments/verbalizer_coverage"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
QUANTIZATION = None  # 量化方法：None/awq/fp8
# ===================================================================


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85, quantization: str = None):
    """加载vLLM模型和tokenizer（复用generate_adjective_c_r_vllm逻辑）"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
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


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（去重）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return list(dict.fromkeys(token_ids))


def build_chat_messages(template, instruction, content, adj, adj_definition=""):
    """
    根据模板类型构建Chat Template的messages列表。
    与 generate_adjective_c_r_vllm.py 中的模板构建保持一致。
    """
    if template == "binary":
        user_content = f"文本内容：{content}\n形容词：「{adj}」描述是否准确？回答： "
    elif template == "likert":
        user_content = f"文本内容：{content}\n形容词：{adj}\n回答： "
    elif template == "ICL":
        user_content = (
            f"文本内容：{content}\n"
            f"形容词「{adj}」的定义：{adj_definition}\n"
            f"根据上述定义，该文本是否表现出该形容词「{adj}」所描述的特征？回答： "
        )
    else:
        raise ValueError(f"不支持的模板类型: {template}")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def analyze_verbalizer_coverage(
    text_content,
    template,
    adjective_path,
    tokenizer,
    llm_model,
    output_dir: Path,
    model_name: str,
):
    """
    对单条文本遍历所有形容词，使用 vLLM 计算 verbalizer 概率总和并可视化。
    """
    # 根据模板类型定义 verbalizer token 和提示词指令
    if template in ["binary", "ICL"]:
        affirmative_tokens = ["是", " 是", "Yes", " Yes", "yes", " yes"]
        negative_tokens = ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer)
        negative_ids = get_first_token_ids(negative_tokens, tokenizer)

        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            definition = pd.read_csv(adjective_path)["definition"].tolist()

    elif template == "likert":
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

    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_en_list = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adjectives)

    # vLLM采样配置
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20
    )

    # 存储结果
    results = []

    # 构建所有提示词
    prompts = []
    for index, adj in enumerate(adjectives):
        if template == "ICL":
            messages = build_chat_messages(template, instruction, text_content, adj, definition[index])
        else:
            messages = build_chat_messages(template, instruction, text_content, adj)

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt_text)

    # 批量推理（vLLM自动处理批量化）
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        # 提取首token的logprobs
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0]

        # 转换为概率字典
        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        if template in ["binary", "ICL"]:
            pos_prob = sum(probs_dict.get(tid, 0.0) for tid in affirmative_ids)
            neg_prob = sum(probs_dict.get(tid, 0.0) for tid in negative_ids)
            total_prob = pos_prob + neg_prob
            results.append({
                "index": adj_idx,
                "adjective_en": adj_en_list[adj_idx],
                "adjective_cn": adjectives[adj_idx],
                "pos_prob": round(pos_prob, 6),
                "neg_prob": round(neg_prob, 6),
                "total_prob": round(total_prob, 6),
            })
        elif template == "likert":
            level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]
            total_prob = sum(level_probs)
            # 计算概率最高的数字分数（1-5映射到0.0-1.0）
            max_level_idx = level_probs.index(max(level_probs))
            max_level_score = max_level_idx / 4.0  # 0, 0.25, 0.5, 0.75, 1.0
            results.append({
                "index": adj_idx,
                "adjective_en": adj_en_list[adj_idx],
                "adjective_cn": adjectives[adj_idx],
                "level_1_prob": round(level_probs[0], 6),
                "level_2_prob": round(level_probs[1], 6),
                "level_3_prob": round(level_probs[2], 6),
                "level_4_prob": round(level_probs[3], 6),
                "level_5_prob": round(level_probs[4], 6),
                "total_prob": round(total_prob, 6),
                "max_level_score": round(max_level_score, 6),
            })

    # 保存JSON数据
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_text = text_content[:20].replace("\\", "").replace("/", "").replace(" ", "_")
    json_path = output_dir / f"{template}_{safe_text}_{model_name}_vllm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "template": template,
            "text_content": text_content,
            "num_adjectives": len(adjectives),
            "statistics": {
                "mean_total_prob": round(sum(r["total_prob"] for r in results) / len(results), 6),
                "min_total_prob": round(min(r["total_prob"] for r in results), 6),
                "max_total_prob": round(max(r["total_prob"] for r in results), 6),
            },
            "data": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"数据已保存: {json_path}")

    # 绘制图表
    fig, ax = plt.subplots(figsize=(16, 6))
    x = [r["index"] for r in results]

    if template in ["binary", "ICL"]:
        pos_probs = [r["pos_prob"] for r in results]
        neg_probs = [r["neg_prob"] for r in results]
        total_probs = [r["total_prob"] for r in results]

        ax.plot(x, pos_probs, label="pos_prob (肯定词概率)", color="green", alpha=0.7, linewidth=0.8)
        ax.plot(x, neg_probs, label="neg_prob (否定词概率)", color="red", alpha=0.7, linewidth=0.8)
        ax.plot(x, total_probs, label="total_prob (verbalizer总概率)", color="blue", alpha=0.9, linewidth=1.2)

        # 添加均值参考线
        mean_total = sum(total_probs) / len(total_probs)
        ax.axhline(y=mean_total, color="blue", linestyle="--", alpha=0.5, label=f"total均值: {mean_total:.3f}")

    elif template == "likert":
        total_probs = [r["total_prob"] for r in results]
        max_level_scores = [r["max_level_score"] for r in results]
        ax.plot(x, total_probs, label="total_prob (Likert verbalizer总概率)", color="blue", alpha=0.9, linewidth=1.2)
        ax.plot(x, max_level_scores, label="max_level_score (概率最高数字分数)", color="orange", alpha=0.8, linewidth=1.0, linestyle="--")

        mean_total = sum(total_probs) / len(total_probs)
        ax.axhline(y=mean_total, color="blue", linestyle="--", alpha=0.5, label=f"total均值: {mean_total:.3f}")

    ax.set_xlabel("形容词索引", fontsize=12)
    ax.set_ylabel("概率", fontsize=12)
    ax.set_title(
        f"Verbalizer覆盖率分析（vLLM）\n模型: {model_name} | 模板: {template} | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, len(adjectives) - 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 在底部添加形容词名称（稀疏显示，避免重叠）
    tick_step = max(1, len(adjectives) // 20)
    tick_positions = list(range(0, len(adjectives), tick_step))
    tick_labels = [adjectives[i] if i < len(adjectives) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    png_path = output_dir / f"{template}_{safe_text}_{model_name}_vllm.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {png_path}")
    plt.close()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("Verbalizer覆盖率统计摘要")
    print("=" * 60)
    print(f"模板类型: {template}")
    print(f"形容词数量: {len(adjectives)}")
    print(f"total_prob 均值: {sum(r['total_prob'] for r in results) / len(results):.4f}")
    print(f"total_prob 最小值: {min(r['total_prob'] for r in results):.4f}")
    print(f"total_prob 最大值: {max(r['total_prob'] for r in results):.4f}")
    if template in ["binary", "ICL"]:
        print(f"pos_prob 均值: {sum(r['pos_prob'] for r in results) / len(results):.4f}")
        print(f"neg_prob 均值: {sum(r['neg_prob'] for r in results) / len(results):.4f}")
    print("=" * 60)

    return results


def main():
    config = MLPConfig()
    output_dir = config.base_path / OUTPUT_DIR

    print("\n" + "=" * 60)
    print("Verbalizer覆盖率分析（vLLM版本）")
    print("=" * 60)
    print(f"模型名称: {MODEL_NAME}")
    print(f"提示词模板: {PROMPT_TEMPLATE}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"量化方法: {QUANTIZATION if QUANTIZATION else '无量化'}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")

    tokenizer, llm_model = load_vllm_model(config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION, QUANTIZATION)
    analyze_verbalizer_coverage(
        text_content=TEXT_CONTENT,
        template=PROMPT_TEMPLATE,
        adjective_path=config.adjective_path,
        tokenizer=tokenizer,
        llm_model=llm_model,
        output_dir=output_dir,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    main()
