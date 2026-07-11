"""3级行为锚定Likert Verbalizer覆盖率全景分析工具（全形容词扫描，vLLM版本）

【定位】
本脚本是 generate_3level_concept_vllm.py 的"全形容词切片"评估工具。
对"一条固定文本 + 全部形容词"进行扫描，评估3级verbalizer在整个形容词词典上的覆盖能力。

【核心功能】
1. 对单条文本遍历所有形容词，提取P(1)/P(2)/P(3)概率和P(3)归一化分数
2. 可视化3级概率分布和概念激活度
3. 统计verbalizer覆盖率、概念激活率、3级分布特征

【输出】
1. 可视化图表（PNG）：3级概率堆叠图 + P(3)分数折线图
2. JSON 数据文件：每个形容词的详细概率数据 + 统计摘要

【使用方法】
1. 修改下方 CONFIG 区域的变量
2. 运行：python scripts/inspect_3level_coverage_vllm.py
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
TEXT_CONTENT = "那些蛆虫就该被消灭"

# 输出目录（相对于项目根目录）
OUTPUT_DIR = "experiments/verbalizer_coverage"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85
# ===================================================================


# 模型加载配置表（与 generate_3level_concept_vllm.py 保持一致）
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen2.5-14B-Instruct": {
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
    """加载vLLM模型和tokenizer（复用generate_3level_concept_vllm逻辑）"""
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
    print(f"  量化方式: {quantization if quantization else '无量化'}")
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


# system instruction（与generate_3level_concept_vllm.py一致）
SYSTEM_INSTRUCTION = (
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


def build_chat_messages(content, adj, adj_definition=None):
    """构建3级行为锚定Likert的Chat Template messages列表。
    逻辑与 generate_3level_concept_vllm.py 中的模板构建保持一致。
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"概念：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"说话者与\"{adj}\"的关系级别（1=不相关 2=涉及 3=实施）：回答：")
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    return messages


def analyze_3level_coverage(
    text_content,
    adjective_path,
    tokenizer,
    llm_model,
    output_dir: Path,
    model_name: str,
    is_qwen3=False,
    prompt_suffix="",
):
    """对单条文本遍历所有形容词，使用 vLLM 计算3级概率分布和P(3)分数并可视化。"""
    # 3级Likert verbalizer token id
    level_tokens = ["1", "2", "3"]
    level_ids = get_first_token_ids(level_tokens, tokenizer)

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

    # 存储结果
    results = []
    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0]

        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        level_probs = [probs_dict.get(tid, 0.0) for tid in level_ids]
        verbalizer_total = sum(level_probs)
        score = level_probs[2] / (verbalizer_total + 1e-8)

        # 双维度分解
        relevance = (level_probs[1] + level_probs[2]) / (verbalizer_total + 1e-8)
        enactment = level_probs[2] / (level_probs[1] + level_probs[2] + 1e-8)

        results.append({
            "index": adj_idx,
            "adjective_en": adj_en_list[adj_idx],
            "adjective_cn": adjectives[adj_idx],
            "p1_not_related": round(level_probs[0], 6),
            "p2_involved": round(level_probs[1], 6),
            "p3_enacting": round(level_probs[2], 6),
            "verbalizer_total": round(verbalizer_total, 6),
            "action_score": round(score, 6),
            "relevance": round(relevance, 6),
            "enactment": round(enactment, 6),
        })

    # 保存JSON数据
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_text = text_content[:20].replace("\\", "").replace("/", "").replace(" ", "_")
    json_path = output_dir / f"3level_{safe_text}_{model_name}_vllm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "template": "3level",
            "text_content": text_content,
            "num_adjectives": len(adjectives),
            "statistics": {
                "mean_verbalizer_total": round(sum(r["verbalizer_total"] for r in results) / len(results), 6),
                "min_verbalizer_total": round(min(r["verbalizer_total"] for r in results), 6),
                "mean_p1": round(sum(r["p1_not_related"] for r in results) / len(results), 6),
                "mean_p2": round(sum(r["p2_involved"] for r in results) / len(results), 6),
                "mean_p3": round(sum(r["p3_enacting"] for r in results) / len(results), 6),
                "mean_action_score": round(sum(r["action_score"] for r in results) / len(results), 6),
            },
            "data": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"数据已保存: {json_path}")

    # 绘制图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    x = [r["index"] for r in results]
    p1 = [r["p1_not_related"] for r in results]
    p2 = [r["p2_involved"] for r in results]
    p3 = [r["p3_enacting"] for r in results]
    scores = [r["action_score"] for r in results]
    relevances = [r["relevance"] for r in results]
    enactments = [r["enactment"] for r in results]

    # 上图：3级概率堆叠柱状图
    ax1.bar(x, p1, label="P(1) 不相关", color="#2196F3", alpha=0.8, width=1.0)
    ax1.bar(x, p2, bottom=p1, label="P(2) 涉及", color="#FF9800", alpha=0.8, width=1.0)
    ax1.bar(x, p3, bottom=[a+b for a, b in zip(p1, p2)], label="P(3) 实施", color="#F44336", alpha=0.8, width=1.0)
    ax1.set_ylabel("概率", fontsize=12)
    ax1.set_title(
        f"3级行为锚定 Verbalizer覆盖率分析（vLLM）\n模型: {model_name} | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 中图：P(3)归一化分数 + 0.5参考线
    ax2.plot(x, scores, label="P(3)归一化分数 (实施度)", color="#F44336", alpha=0.9, linewidth=1.2)
    mean_score = sum(scores) / len(scores)
    ax2.axhline(y=mean_score, color="#F44336", linestyle="--", alpha=0.5, label=f"均值: {mean_score:.3f}")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="0.5参考线")
    ax2.set_ylabel("P(3)分数", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # 下图：双维度分解 - 相关性 + 实施度
    ax3.plot(x, relevances, label="相关性 (P(2)+P(3))", color="#4CAF50", alpha=0.9, linewidth=1.2)
    ax3.plot(x, enactments, label="实施度 (P(3)/(P(2)+P(3)))", color="#F44336", alpha=0.9, linewidth=1.2)
    ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="0.5参考线")
    ax3.set_xlabel("形容词索引", fontsize=12)
    ax3.set_ylabel("概率", fontsize=12)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # 在底部添加形容词名称
    tick_step = max(1, len(adjectives) // 20)
    tick_positions = list(range(0, len(adjectives), tick_step))
    tick_labels = [adjectives[i] if i < len(adjectives) else "" for i in tick_positions]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    png_path = output_dir / f"3level_{safe_text}_{model_name}_vllm.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {png_path}")
    plt.close()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("3级行为锚定 Verbalizer覆盖率统计摘要")
    print("=" * 60)
    print(f"形容词数量: {len(adjectives)}")
    print(f"Verbalizer总概率均值: {sum(r['verbalizer_total'] for r in results) / len(results):.4f}")
    print(f"P(1)不相关 均值: {sum(p1) / len(p1):.4f}")
    print(f"P(2)涉及   均值: {sum(p2) / len(p2):.4f}")
    print(f"P(3)实施   均值: {sum(p3) / len(p3):.4f}")
    print(f"P(3)归一化分数均值: {sum(scores) / len(scores):.4f}")

    # P(3) > 0.5 的概念（LLM判断说话者正在实施的行为）
    active_concepts = [r for r in results if r["action_score"] > 0.5]
    print(f"\nP(3)>0.5 的概念数（说话者正在实施）: {len(active_concepts)}/{len(results)}")
    active_concepts.sort(key=lambda r: -r["action_score"])
    for r in active_concepts[:10]:
        print(f"  {r['adjective_cn']:<20} P(3)={r['action_score']:.4f}  P(2)={r['p2_involved']:.4f}  P(1)={r['p1_not_related']:.4f}")

    # P(2) > 0.5 的概念（LLM判断涉及但未实施）
    involved_concepts = [r for r in results if r["p2_involved"] > r["p3_enacting"] and r["p2_involved"] > r["p1_not_related"]]
    print(f"\nP(2)最高的概念数（涉及但未实施）: {len(involved_concepts)}/{len(results)}")
    involved_concepts.sort(key=lambda r: -r["p2_involved"])
    for r in involved_concepts[:10]:
        print(f"  {r['adjective_cn']:<20} P(2)={r['p2_involved']:.4f}  P(3)={r['p3_enacting']:.4f}  P(1)={r['p1_not_related']:.4f}")

    print("=" * 60)

    return results


def main():
    config = MLPConfig()
    output_dir = config.base_path / OUTPUT_DIR

    print("\n" + "=" * 60)
    print("3级行为锚定 Verbalizer覆盖率分析（vLLM版本）")
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

    analyze_3level_coverage(
        text_content=TEXT_CONTENT,
        adjective_path=config.adjective_path,
        tokenizer=tokenizer,
        llm_model=llm_model,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix,
    )


if __name__ == "__main__":
    main()
