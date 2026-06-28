"""意图概念向量 Verbalizer覆盖率全景分析工具（全形容词扫描，vLLM版本）

【定位】
本脚本是 generate_adjective_intent_vllm.py 的"全形容词切片"评估工具。
generate_adjective_intent_vllm.py 负责为数据集中所有文本、所有形容词批量生成意图概念向量；
inspect_prompt_template_intent_vllm.py 负责在单样本级别（一个文本 + 一个形容词）调试提示词和 verbalizer；
而本脚本则对"一条固定文本 + 全部形容词"进行扫描，评估该提示词模板和 verbalizer 词表在整个形容词词典上的覆盖能力是否稳定。

【核心功能】
对单条文本遍历所有形容词，使用 vLLM 推理并提取 verbalizer token 的概率总和。

【与 generate_adjective_intent_vllm.py 的关系】
- 本脚本的提示词构建逻辑、verbalizer 词表（"1"~"3"）、分数计算逻辑与
  generate_adjective_intent_vllm.py 完全一致。

【输出】
1. 可视化图表（PNG）：横轴为形容词索引，纵轴为概率值
2. JSON 数据文件：每个形容词的详细概率数据 + 统计摘要

【使用方法】
1. 修改下方 CONFIG 区域的变量（模型名、文本内容等）
2. 运行：python scripts/inspect_verbalizer_coverage_intent_vllm.py
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
MODEL_NAME = "glm-4-9b-chat"  # models目录下的模型文件夹名

# 文本内容（直接修改即可）
TEXT_CONTENT = "什么被害妄想猎巫man"

# 输出目录（相对于项目根目录）
OUTPUT_DIR = "experiments/verbalizer_coverage"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
# ===================================================================


# 模型加载配置表（与 generate_adjective_intent_vllm.py 保持一致）
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
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置。未知模型将直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer（复用generate_adjective_intent_vllm逻辑）"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
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
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f" 量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（去重）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")

    return list(dict.fromkeys(token_ids))


# system instruction（与generate_adjective_intent_vllm.py一致）
SYSTEM_INSTRUCTION = (
    "你是一位语用分析专家，擅长识别文本中作者的真实意图。\n"
    "请综合考虑文本的字面意思和隐含语义（比喻、谐音、反讽、文化隐喻等），"
    "判断作者在文本中是否以该形容词所描述的方式表达其态度。\n"
    "用1到3的数字回答，1表示作者仅在讨论、引用、反对或客观陈述相关话题"
    "而未以该方式表达态度，2表示难以明确判断，"
    "3表示作者以该形容词所描述的方式表达其态度，包括以隐含、暗示或反讽等方式。只回答一个数字。"
)


def build_chat_messages(content, adj, adj_definition=None):
    """构建意图判断的Chat Template messages。
    逻辑与 generate_adjective_intent_vllm.py 中的模板构建保持一致。
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"作者是否以\"{adj}\"所描述的方式表达其态度？回答： ")
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
    template="intent",
):
    """对单条文本遍历所有形容词，使用 vLLM 计算 verbalizer 概率总和并可视化。

    3级意图verbalizer：1=讨论/引用/反对, 2=模糊, 3=表达
    """
    # 意图verbalizer token（3级）
    intent_tokens = ["1", "2", "3"]
    intent_ids = get_first_token_ids(intent_tokens, tokenizer)

    # 加载形容词词典（含定义）
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_en_list = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adjectives)
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)

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
    for adj, adj_def in zip(adjectives, adj_definitions):
        messages = build_chat_messages(text_content, adj, adj_def)

        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # 批量推理
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        # 提取首token的logprobs
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0]

        # 转换为概率字典
        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        # 提取3级意图概率
        level_probs = [probs_dict.get(tid, 0.0) for tid in intent_ids]
        total_prob = sum(level_probs)
        # 概率最高数字映射到分数（3级：0, 0.5, 1.0）
        max_level_idx = level_probs.index(max(level_probs))
        max_level_score = max_level_idx / 2.0  # 0, 0.5, 1.0
        # 加权期望意图分数（与generate_adjective_intent_vllm.py一致）
        weights = [0.0, 0.5, 1.0]
        total_prob_eps = total_prob + 1e-8
        intent_score = sum(w * p for w, p in zip(weights, level_probs)) / total_prob_eps
        results.append({
            "index": adj_idx,
            "adjective_en": adj_en_list[adj_idx],
            "adjective_cn": adjectives[adj_idx],
            "level_1_prob": round(level_probs[0], 6),
            "level_2_prob": round(level_probs[1], 6),
            "level_3_prob": round(level_probs[2], 6),
            "total_prob": round(total_prob, 6),
            "max_level_score": round(max_level_score, 6),
            "intent_score": round(intent_score, 6),
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

    total_probs = [r["total_prob"] for r in results]
    max_level_scores = [r["max_level_score"] for r in results]
    intent_scores = [r["intent_score"] for r in results]
    ax.plot(x, total_probs, label="total_prob (意图verbalizer总概率)", color="blue", alpha=0.9, linewidth=1.2)
    ax.plot(x, max_level_scores, label="max_level_score (概率最高数字分数)", color="orange", alpha=0.8, linewidth=1.0, linestyle="--")
    ax.plot(x, intent_scores, label="intent_score (加权期望分数)", color="green", alpha=0.8, linewidth=1.0, linestyle="-.")

    mean_total = sum(total_probs) / len(total_probs)
    ax.axhline(y=mean_total, color="blue", linestyle="--", alpha=0.5, label=f"total均值: {mean_total:.3f}")

    ax.set_xlabel("形容词索引", fontsize=12)
    ax.set_ylabel("概率", fontsize=12)
    ax.set_title(
        f"意图Verbalizer覆盖率分析（vLLM）\n模型: {model_name} | 模板: intent | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, len(adjectives) - 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 在底部添加形容词名称
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
    print("意图Verbalizer覆盖率统计摘要")
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
    print("意图概念向量 Verbalizer覆盖率分析（vLLM版本）")
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
        template="intent",
    )


if __name__ == "__main__":
    main()
