"""说话者视角 Verbalizer覆盖率全景分析工具（全形容词扫描，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm_speaker.py 的"全形容词切片"评估工具。
对"一条固定文本 + 全部形容词"进行扫描，评估说话者视角提示词模板
在整个形容词词典上的覆盖能力是否稳定。

与 Likert 版（inspect_verbalizer_coverage_vllm.py）的区别：
  - System instruction: 语言行为分析专家 vs 语言分析专家
  - User question: "说话者在多大程度上表现出X的行为特征" vs "文本在多大程度上体现了X的特征"

【使用方法】
修改下方 CONFIG 区域的变量，然后运行：
python scripts/inspect_verbalizer_coverage_vllm_speaker.py
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "glm-4-9b-chat"

# 文本内容
TEXT_CONTENT = "他只会辱骂诬陷"

# 输出目录
OUTPUT_DIR = "experiments/verbalizer_coverage"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85
# ===================================================================


# 模型加载配置表（与 generate_adjective_c_r_vllm_speaker.py 保持一致）
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",
        "is_qwen3": True,
        "is_multimodal": True,
        "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "\n",
    },
    "deepseek-llm-7b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Baichuan2-7B-Chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3-8B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
}


def get_model_loading_config(model_name: str) -> dict:
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
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


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))


# ★说话者视角 System instruction
SYSTEM_INSTRUCTION = (
    "你是一位语言行为分析专家，擅长识别说话者的表达行为方式。\n"
    "在评估时，请重点关注说话者（即文本的发出者）是否在主动表现出该形容词所描述的行为特征，"
    "而非仅仅描述或讨论该特征。\n"
    "例如：\"他只会辱骂诬陷\"中说话者是在描述他人的侮辱行为，而非自己表现出侮辱性；"
    "\"你个废物\"中说话者是在主动表现出侮辱性的行为。\n"
    "请用1到5的数字评估说话者的表现程度，1表示完全没有表现出，5表示强烈表现出。只回答一个数字。"
)


# ★说话者视角 Prompt构建
def build_chat_messages(content, adj, adj_definition=None):
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本的说话者在多大程度上表现出\"{adj}\"的行为特征？回答： ")
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    return messages


def analyze_verbalizer_coverage(
    text_content,
    adjective_path,
    tokenizer,
    llm_model,
    output_dir: Path,
    model_name: str,
    is_qwen3=False,
    prompt_suffix="",
    template="speaker",
):
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_en_list = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adjectives)
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []

    prompts = []
    for adj, adj_def in zip(adjectives, adj_definitions):
        messages = build_chat_messages(text_content, adj, adj_def)
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0]

        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]
        total_prob = sum(level_probs)
        max_level_idx = level_probs.index(max(level_probs))
        max_level_score = max_level_idx / 4.0
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        total_prob_eps = total_prob + 1e-8
        likert_score = sum(w * p for w, p in zip(weights, level_probs)) / total_prob_eps
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
            "likert_score": round(likert_score, 6),
        })

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

    total_probs = [r["total_prob"] for r in results]
    max_level_scores = [r["max_level_score"] for r in results]
    likert_scores = [r["likert_score"] for r in results]
    ax.plot(x, total_probs, label="total_prob (Likert verbalizer总概率)", color="blue", alpha=0.9, linewidth=1.2)
    ax.plot(x, max_level_scores, label="max_level_score (概率最高数字分数)", color="orange", alpha=0.8, linewidth=1.0, linestyle="--")
    ax.plot(x, likert_scores, label="likert_score (加权期望分数)", color="green", alpha=0.8, linewidth=1.0, linestyle="-.")

    mean_total = sum(total_probs) / len(total_probs)
    ax.axhline(y=mean_total, color="blue", linestyle="--", alpha=0.5, label=f"total均值: {mean_total:.3f}")

    ax.set_xlabel("形容词索引", fontsize=12)
    ax.set_ylabel("概率", fontsize=12)
    ax.set_title(
        f"说话者视角 Verbalizer覆盖率分析（vLLM）\n模型: {model_name} | 模板: speaker | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, len(adjectives) - 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

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
    print("说话者视角 Verbalizer覆盖率统计摘要")
    print("=" * 60)
    print(f"模板类型: {template}")
    print(f"形容词数量: {len(adjectives)}")
    print(f"total_prob 均值: {sum(r['total_prob'] for r in results) / len(results):.4f}")
    print(f"total_prob 最小值: {min(r['total_prob'] for r in results):.4f}")
    print(f"total_prob 最大值: {max(r['total_prob'] for r in results):.4f}")
    print("=" * 60)

    return results


def main():
    config = MLPConfig()
    output_dir = config.base_path / OUTPUT_DIR

    print("\n" + "=" * 60)
    print("说话者视角 Verbalizer覆盖率分析（vLLM版本）")
    print("=" * 60)
    print(f"模型名称: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION)
    if qwen3_flag:
        print(f"检测到Qwen3+模型({MODEL_NAME})，已禁用思考模式(enable_thinking=False)")
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({MODEL_NAME})需要追加prompt后缀: {repr(prompt_suffix)}")

    analyze_verbalizer_coverage(
        text_content=TEXT_CONTENT,
        adjective_path=config.adjective_path,
        tokenizer=tokenizer,
        llm_model=llm_model,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix,
        template="speaker",
    )


if __name__ == "__main__":
    main()
