"""类型感知提示词Verbalizer覆盖率验证工具（vLLM版本）

【定位】
本脚本是 generate_typed_concept_vllm.py 的"面级"前置验证工具。
在全量概念向量生成之前，对一条文本 + 全部概念进行快速扫描，
验证7种提示词类型的verbalizer覆盖率是否达标。

【核心功能】
对单条（或少数几条）文本遍历所有概念，按7种提示词类型分组统计：
1. 各类型verbalizer覆盖率（LLM首token是否落在verbalizer词表中）
2. 各类型概率分布（P(1)/P(2)/P(3)均值和范围）
3. 逐概念覆盖率明细
4. 低覆盖概念警告

【与 generate_typed_concept_vllm.py 的关系】
- 本脚本的提示词构建、verbalizer词表、概率计算逻辑与生成脚本完全一致
- 用于"面级"前置验证：确认改进后的提示词在整个概念词典上表现稳定
- 验证通过后再运行全量生成，避免浪费计算资源

【使用方法】
    python scripts/inspect_typed_verbalizer_coverage_vllm.py
"""

import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from tqdm import tqdm

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from generate_typed_concept_vllm import (
    PROMPT_REGISTRY, THREE_LEVEL_TYPES, BINARY_TYPES,
    load_concepts, build_prompt,
    MODEL_LOADING_CONFIG, get_model_loading_config, load_vllm_model,
)
from configs.MLP_config import MLPConfig


# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "glm-4-9b-chat"

# 测试文本（建议选1条有毒+1条无毒）
TEST_TEXTS = [
    ("什么被害妄想猎巫man", 1),
    ("今天天气真好适合出门散步", 0),
]

# 概念词典
ADJECTIVE_NAME = "toxic_adjectives_v4.csv"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85
# ===================================================================


def analyze_single_text(text_content, text_label, concepts, tokenizer, llm_model,
                        is_qwen3, prompt_suffix):
    """对单条文本遍历所有概念，返回每个概念的verbalizer概率数据。"""
    sampling_params = __import__("vllm").SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 构建所有概念的prompt
    prompts = []
    prompt_meta = []  # (concept_idx, ptype, verbalizer)

    for ci, concept in enumerate(concepts):
        system_prompt, user_prompt = build_prompt(text_content, concept)
        ptype = concept["prompt_template"]
        verbalizer = PROMPT_REGISTRY[ptype]["verbalizer"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix

        prompts.append(prompt_text)
        prompt_meta.append((ci, ptype, verbalizer))

    # 批量推理
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    # 解析结果
    results = []
    for pi, output in enumerate(outputs):
        ci, ptype, verbalizer = prompt_meta[pi]

        # 提取首token logprobs
        token_logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            first_lp = output.outputs[0].logprobs[0]
            for token_id, logprob_info in first_lp.items():
                token_text = logprob_info.decoded_token.strip()
                if token_text in verbalizer:
                    token_logprobs[token_text] = logprob_info.logprob

        # 计算概率（与generate脚本一致：softmax归一化）
        probs = {}
        if token_logprobs:
            max_logprob = max(token_logprobs.values())
            exp_sum = sum(np.exp(lp - max_logprob) for lp in token_logprobs.values())
            for v in verbalizer:
                if v in token_logprobs:
                    probs[v] = float(np.exp(token_logprobs[v] - max_logprob) / exp_sum)
                else:
                    probs[v] = 0.0
        else:
            for v in verbalizer:
                probs[v] = 1.0 / len(verbalizer)

        total_prob = sum(probs.values())
        results.append({
            "concept_idx": ci,
            "name": concepts[ci]["name"],
            "ptype": ptype,
            "verbalizer": verbalizer,
            "probs": probs,
            "total_prob": total_prob,
        })

    return results


def main():
    config = MLPConfig()
    adj_path = config.raw_data_path / "adjective" / ADJECTIVE_NAME

    # 加载概念
    concepts = load_concepts(adj_path)
    print(f"\n概念总数: {len(concepts)}")
    type_counts = defaultdict(int)
    for c in concepts:
        type_counts[c["prompt_template"]] += 1
    for ptype, count in sorted(type_counts.items()):
        v_type = "3级" if ptype in THREE_LEVEL_TYPES else "二元"
        print(f"  {ptype}: {count}概念 ({v_type})")

    # 加载模型
    tokenizer, llm_model, is_qwen3 = load_vllm_model(
        config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION
    )
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")

    if is_qwen3:
        print(f"检测到Qwen3+模型({MODEL_NAME})，已禁用思考模式")
    if prompt_suffix:
        print(f"检测到模型({MODEL_NAME})需要追加prompt后缀: {repr(prompt_suffix)}")

    # 打印verbalizer token ids
    print("\nVerbalizer token映射:")
    for ptype in PROMPT_REGISTRY:
        verbalizer = PROMPT_REGISTRY[ptype]["verbalizer"]
        ids = []
        for v in verbalizer:
            encoded = tokenizer.encode(v, add_special_tokens=False)
            if encoded:
                ids.append((v, encoded[0]))
        print(f"  {ptype}: {ids}")

    # 对每条文本进行验证
    all_results = {}
    for text_content, text_label in TEST_TEXTS:
        label_str = "有毒" if text_label == 1 else "无毒"
        print(f"\n{'='*60}")
        print(f"验证文本: \"{text_content}\" ({label_str})")
        print(f"{'='*60}")

        results = analyze_single_text(
            text_content, text_label, concepts,
            tokenizer, llm_model, is_qwen3, prompt_suffix
        )
        all_results[(text_content, text_label)] = results

    # ============================================================
    # 生成报告
    # ============================================================
    report_lines = []
    def w(text=""):
        report_lines.append(text)
        print(text)

    w("类型感知提示词Verbalizer覆盖率验证报告")
    w(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"模型: {MODEL_NAME}")
    w(f"词典: {ADJECTIVE_NAME} ({len(concepts)}概念)")
    w(f"验证文本数: {len(TEST_TEXTS)}")
    w("")

    # ============================================================
    # 分析1: 各类型Verbalizer覆盖率（按文本分组）
    # ============================================================
    w("=" * 70)
    w("分析1: 各类型Verbalizer覆盖率")
    w("=" * 70)

    for (text_content, text_label), results in all_results.items():
        label_str = "有毒" if text_label == 1 else "无毒"
        w(f"\n  文本: \"{text_content}\" ({label_str})")

        for ptype in PROMPT_REGISTRY:
            ptype_results = [r for r in results if r["ptype"] == ptype]
            if not ptype_results:
                continue

            verbalizer = PROMPT_REGISTRY[ptype]["verbalizer"]
            v_type = "3级" if ptype in THREE_LEVEL_TYPES else "二元"

            total = len(ptype_results)
            covered = sum(1 for r in ptype_results if r["total_prob"] > 0.01)
            coverage = covered / total * 100 if total > 0 else 0
            mean_total = np.mean([r["total_prob"] for r in ptype_results])
            min_total = np.min([r["total_prob"] for r in ptype_results])

            w(f"\n    {ptype} ({v_type}, {total}概念):")
            w(f"      覆盖率: {coverage:.1f}% ({covered}/{total})")
            w(f"      总概率: 均值={mean_total:.4f}, 最小={min_total:.4f}")

            for v in verbalizer:
                vals = [r["probs"].get(v, 0.0) for r in ptype_results]
                w(f"      P({v}): 均值={np.mean(vals):.4f}, 范围=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    # ============================================================
    # 分析2: 逐概念覆盖率明细（按类型分组）
    # ============================================================
    w("\n" + "=" * 70)
    w("分析2: 逐概念覆盖率明细")
    w("=" * 70)

    for (text_content, text_label), results in all_results.items():
        label_str = "有毒" if text_label == 1 else "无毒"
        w(f"\n  文本: \"{text_content}\" ({label_str})")

        for ptype in PROMPT_REGISTRY:
            ptype_results = [r for r in results if r["ptype"] == ptype]
            if not ptype_results:
                continue

            verbalizer = PROMPT_REGISTRY[ptype]["verbalizer"]
            w(f"\n    {ptype} ({len(ptype_results)}概念):")
            header = f"    {'概念名':<20} {'总概率':>8}"
            for v in verbalizer:
                header += f" {'P('+v+')':>8}"
            w(header)
            w("    " + "-" * (28 + 9 * len(verbalizer)))

            for r in ptype_results:
                line = f"    {r['name']:<20} {r['total_prob']:>8.4f}"
                for v in verbalizer:
                    line += f" {r['probs'].get(v, 0.0):>8.4f}"
                w(line)

    # ============================================================
    # 分析3: 有毒vs无毒概率分布对比（仅当有多条文本时）
    # ============================================================
    if len(all_results) >= 2:
        w("\n" + "=" * 70)
        w("分析3: 有毒vs无毒主信号对比")
        w("=" * 70)

        toxic_results = None
        nontoxic_results = None
        for (text_content, text_label), results in all_results.items():
            if text_label == 1:
                toxic_results = results
            else:
                nontoxic_results = results

        if toxic_results and nontoxic_results:
            w(f"\n  {'类型':<16} {'主信号':>8} {'有毒均值':>10} {'无毒均值':>10} {'差值':>10}")
            w("  " + "-" * 58)

            for ptype in PROMPT_REGISTRY:
                t_results = [r for r in toxic_results if r["ptype"] == ptype]
                nt_results = [r for r in nontoxic_results if r["ptype"] == ptype]
                if not t_results or not nt_results:
                    continue

                signal_name = "P(2)" if ptype in BINARY_TYPES else "P(3)"
                signal_key = "2" if ptype in BINARY_TYPES else "3"

                t_mean = np.mean([r["probs"].get(signal_key, 0.0) for r in t_results])
                nt_mean = np.mean([r["probs"].get(signal_key, 0.0) for r in nt_results])
                diff = t_mean - nt_mean

                w(f"  {ptype:<16} {signal_name:>8} {t_mean:>10.4f} {nt_mean:>10.4f} {diff:>+10.4f}")

    # ============================================================
    # 分析4: 低覆盖概念警告
    # ============================================================
    w("\n" + "=" * 70)
    w("分析4: 低覆盖/异常概念警告")
    w("=" * 70)

    for (text_content, text_label), results in all_results.items():
        label_str = "有毒" if text_label == 1 else "无毒"
        low_coverage = [r for r in results if r["total_prob"] < 0.5]

        if low_coverage:
            w(f"\n  文本\"{text_content}\"({label_str})中总概率<0.5的概念:")
            for r in low_coverage:
                w(f"    {r['name']} ({r['ptype']}): 总概率={r['total_prob']:.4f}")
        else:
            w(f"\n  文本\"{text_content}\"({label_str})中所有概念总概率>=0.5，无异常。")

    # 二元类型P(3)泄漏检测
    for (text_content, text_label), results in all_results.items():
        label_str = "有毒" if text_label == 1 else "无毒"
        for ptype in BINARY_TYPES:
            ptype_results = [r for r in results if r["ptype"] == ptype]
            leak = [r for r in ptype_results if "3" in r["probs"] and r["probs"]["3"] > 0.01]
            if leak:
                w(f"  ⚠ 文本\"{text_content}\"({label_str})中{ptype}类型有{len(leak)}/{len(ptype_results)}概念出现P(3)泄漏")

    # ============================================================
    # 保存报告
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = project_root / "experiments" / f"typed_verbalizer_coverage_{timestamp}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    w(f"\n报告已保存至: {report_path}")

    # ============================================================
    # 保存JSON数据（每个文本的详细概率数据）
    # ============================================================
    json_data = {
        "model_name": MODEL_NAME,
        "adjective_file": ADJECTIVE_NAME,
        "num_concepts": len(concepts),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "texts": [],
    }

    for (text_content, text_label), results in all_results.items():
        text_entry = {
            "content": text_content,
            "toxic": text_label,
            "concepts": [],
        }
        for r in results:
            concept_entry = {
                "name": r["name"],
                "type": r["ptype"],
                "total_prob": round(r["total_prob"], 6),
                "probs": {k: round(v, 6) for k, v in r["probs"].items()},
            }
            text_entry["concepts"].append(concept_entry)
        json_data["texts"].append(text_entry)

    json_path = report_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    w(f"JSON数据已保存至: {json_path}")


if __name__ == "__main__":
    main()
