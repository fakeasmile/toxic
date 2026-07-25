"""Verbalizer覆盖率与信号质量分析脚本。

分析generate_typed_concept_vllm.py输出的概念向量JSON文件，提供7个维度：
1. 各类型Verbalizer覆盖率（逐概念）
2. 有毒/无毒概率分布对比
3. 逐概念SNR排名
4. 3级概念P(2)方向分析
5. 二元类型信号泄漏检测
6. 各类型汇总对比
7. 低覆盖/低SNR警告

不需要vLLM，直接分析已生成的JSON文件。

使用示例：
    python scripts/analyze_verbalizer_coverage.py \
        --concept_file data/processed/TOXICN/glm-4-9b-chat/concept_train_glm-4-9b-chat_typed_v4.json
"""

import argparse
import json
import sys
import io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}


def parse_args():
    parser = argparse.ArgumentParser(description="Verbalizer覆盖率与信号质量分析")
    parser.add_argument("--concept_file", type=str, required=True, help="概念向量JSON文件路径")
    parser.add_argument("--num_samples", type=int, default=0, help="使用前N条样本，0=全量")
    return parser.parse_args()


def compute_snr(toxic_vals, nontoxic_vals):
    if len(toxic_vals) < 2 or len(nontoxic_vals) < 2:
        return 0.0
    diff = abs(np.mean(toxic_vals) - np.mean(nontoxic_vals))
    std_sum = np.std(toxic_vals) + np.std(nontoxic_vals)
    if std_sum < 1e-8:
        return 0.0
    return diff / std_sum


def main():
    args = parse_args()
    report = io.StringIO()

    def w(text=""):
        print(text)
        report.write(text + "\n")

    # 加载数据
    with open(args.concept_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = raw["meta"]
    data = raw["data"]
    if args.num_samples > 0:
        data = data[:args.num_samples]

    concept_names = meta["concept_names"]
    concept_types = meta["concept_types"]
    n_concepts = len(concept_names)
    n_samples = len(data)
    labels = np.array([item["toxic"] for item in data])
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0

    # 按类型分组索引
    type_indices = defaultdict(list)
    for i, ct in enumerate(concept_types):
        type_indices[ct].append(i)

    # 提取分数矩阵
    scores = np.array([item["concept_scores"] for item in data])

    w("=" * 70)
    w("Verbalizer覆盖率与信号质量分析")
    w("=" * 70)
    w(f"概念向量文件: {args.concept_file}")
    w(f"样本数: {n_samples} (有毒={toxic_mask.sum()}, 无毒={nontoxic_mask.sum()})")
    w(f"概念数: {n_concepts}")
    w(f"词典文件: {meta.get('adjective_file', '?')}")
    w(f"生成时间: {meta.get('timestamp', '?')}")

    # ================================================================
    # 分析1: 各类型Verbalizer覆盖率（逐概念）
    # ================================================================
    w("\n" + "=" * 70)
    w("分析1: 逐概念Verbalizer覆盖率")
    w("=" * 70)

    for ptype in ["behavior", "strategy", "evaluation", "discrimination", "neutral", "intent", "effect"]:
        indices = type_indices.get(ptype, [])
        if not indices:
            continue
        v_type = "3级" if ptype in THREE_LEVEL_TYPES else "二元"
        w(f"\n  {ptype}类型 ({len(indices)}概念, {v_type}verbalizer):")
        w(f"  {'概念名':<24} {'覆盖率':>8} {'零概率次数':>10} {'P(1)均值':>10} {'P(2)均值':>10}" +
          (" {'P(3)均值':>10}" if ptype in THREE_LEVEL_TYPES else ""))

        for ci in indices:
            name = concept_names[ci]
            n_covered = 0
            n_zero = 0
            p1_vals, p2_vals, p3_vals = [], [], []
            for item in data:
                lp = item["level_probs"][ci]
                if any(p > 0.01 for p in lp):
                    n_covered += 1
                if all(p < 1e-6 for p in lp):
                    n_zero += 1
                p1_vals.append(lp[0])
                p2_vals.append(lp[1])
                if len(lp) > 2:
                    p3_vals.append(lp[2])

            rate = n_covered / n_samples * 100
            if ptype in THREE_LEVEL_TYPES:
                w(f"  {name:<24} {rate:>7.1f}% {n_zero:>10} {np.mean(p1_vals):>10.4f} {np.mean(p2_vals):>10.4f} {np.mean(p3_vals):>10.4f}")
            else:
                w(f"  {name:<24} {rate:>7.1f}% {n_zero:>10} {np.mean(p1_vals):>10.4f} {np.mean(p2_vals):>10.4f}")

    # ================================================================
    # 分析2: 有毒/无毒概率分布对比
    # ================================================================
    w("\n" + "=" * 70)
    w("分析2: 有毒/无毒概率分布对比（各类型汇总）")
    w("=" * 70)

    w(f"\n  {'类型':<16} {'主信号':>8} {'有毒均值':>10} {'无毒均值':>10} {'差值':>10}")
    w("  " + "-" * 58)

    for ptype in ["behavior", "strategy", "evaluation", "discrimination", "neutral", "intent", "effect"]:
        indices = type_indices.get(ptype, [])
        if not indices:
            continue
        signal_name = "P(2)" if ptype in BINARY_TYPES else "P(3)"
        signal_idx = 1 if ptype in BINARY_TYPES else 2

        toxic_vals = []
        nontoxic_vals = []
        for item in data:
            for ci in indices:
                lp = item["level_probs"][ci]
                if len(lp) > signal_idx:
                    if item["toxic"] == 1:
                        toxic_vals.append(lp[signal_idx])
                    else:
                        nontoxic_vals.append(lp[signal_idx])

        t_mean = np.mean(toxic_vals) if toxic_vals else 0
        nt_mean = np.mean(nontoxic_vals) if nontoxic_vals else 0
        diff = t_mean - nt_mean
        w(f"  {ptype:<16} {signal_name:>8} {t_mean:>10.4f} {nt_mean:>10.4f} {diff:>+10.4f}")

    # ================================================================
    # 分析3: 逐概念SNR排名
    # ================================================================
    w("\n" + "=" * 70)
    w("分析3: 逐概念SNR排名")
    w("=" * 70)

    snr_list = []
    for ci in range(n_concepts):
        t_vals = scores[toxic_mask, ci]
        nt_vals = scores[nontoxic_mask, ci]
        snr = compute_snr(t_vals, nt_vals)
        direction = "→有毒高" if np.mean(t_vals) > np.mean(nt_vals) else "→无毒高"
        snr_list.append((concept_names[ci], concept_types[ci], snr, direction))

    snr_list.sort(key=lambda x: -x[2])

    w(f"\n  SNR Top-20:")
    w(f"  {'排名':>4} {'概念名':<24} {'类型':<14} {'SNR':>8} {'方向':>8}")
    w("  " + "-" * 62)
    for rank, (name, ctype, snr, direction) in enumerate(snr_list[:20], 1):
        w(f"  {rank:>4} {name:<24} {ctype:<14} {snr:>8.4f} {direction:>8}")

    w(f"\n  SNR Bottom-10:")
    w(f"  {'排名':>4} {'概念名':<24} {'类型':<14} {'SNR':>8} {'方向':>8}")
    w("  " + "-" * 62)
    for rank, (name, ctype, snr, direction) in enumerate(snr_list[-10:], n_concepts - 9):
        w(f"  {rank:>4} {name:<24} {ctype:<14} {snr:>8.4f} {direction:>8}")

    # ================================================================
    # 分析4: 3级概念P(2)方向分析
    # ================================================================
    w("\n" + "=" * 70)
    w('分析4: 3级概念P(2)方向分析（"存在但无害"是anti-pattern信号？）')
    w("=" * 70)

    for ptype in ["evaluation", "discrimination", "neutral"]:
        indices = type_indices.get(ptype, [])
        if not indices:
            continue

        reversed_count = 0
        w(f"\n  {ptype}类型 ({len(indices)}概念):")
        w(f"  {'概念名':<24} {'P(2)有毒':>10} {'P(2)无毒':>10} {'差值':>10} {'方向':>10}")
        w("  " + "-" * 68)

        for ci in indices:
            p2_toxic = np.mean([item["level_probs"][ci][1] for item in data if item["toxic"] == 1])
            p2_nontoxic = np.mean([item["level_probs"][ci][1] for item in data if item["toxic"] == 0])
            diff = p2_toxic - p2_nontoxic
            direction = "→有毒高" if diff > 0 else "→无毒高(反转)"
            if diff < 0:
                reversed_count += 1
            w(f"  {concept_names[ci]:<24} {p2_toxic:>10.4f} {p2_nontoxic:>10.4f} {diff:>+10.4f} {direction:>10}")

        w(f"  P(2)反转率: {reversed_count}/{len(indices)} ({reversed_count/len(indices)*100:.1f}%)")

    # ================================================================
    # 分析5: 二元类型信号泄漏检测
    # ================================================================
    w("\n" + "=" * 70)
    w("分析5: 二元类型信号泄漏检测（P(3)是否出现在二元prompt输出中）")
    w("=" * 70)

    for ptype in BINARY_TYPES:
        indices = type_indices.get(ptype, [])
        if not indices:
            continue
        leak_count = 0
        for item in data:
            for ci in indices:
                lp = item["level_probs"][ci]
                if len(lp) > 2:
                    leak_count += 1
        total = n_samples * len(indices)
        w(f"  {ptype}: P(3)泄漏率 = {leak_count}/{total} ({leak_count/total*100:.2f}%)")

    # ================================================================
    # 分析6: 各类型汇总对比
    # ================================================================
    w("\n" + "=" * 70)
    w("分析6: 各类型汇总对比")
    w("=" * 70)

    w(f"\n  {'类型':<16} {'概念数':>6} {'平均SNR':>10} {'主信号差值':>12} {'P(2)反转率':>12}")
    w("  " + "-" * 60)

    for ptype in ["behavior", "strategy", "evaluation", "discrimination", "neutral", "intent", "effect"]:
        indices = type_indices.get(ptype, [])
        if not indices:
            continue

        avg_snr = np.mean([snr for name, ct, snr, _ in snr_list if ct == ptype])

        signal_idx = 1 if ptype in BINARY_TYPES else 2
        t_vals = [item["level_probs"][ci][signal_idx] for item in data for ci in indices if item["toxic"] == 1]
        nt_vals = [item["level_probs"][ci][signal_idx] for item in data for ci in indices if item["toxic"] == 0]
        signal_diff = np.mean(t_vals) - np.mean(nt_vals) if t_vals and nt_vals else 0

        if ptype in THREE_LEVEL_TYPES:
            rev_count = sum(1 for ci in indices
                           if np.mean([item["level_probs"][ci][1] for item in data if item["toxic"] == 0])
                           > np.mean([item["level_probs"][ci][1] for item in data if item["toxic"] == 1]))
            rev_rate = f"{rev_count}/{len(indices)} ({rev_count/len(indices)*100:.0f}%)"
        else:
            rev_rate = "N/A(二元)"

        w(f"  {ptype:<16} {len(indices):>6} {avg_snr:>10.4f} {signal_diff:>+12.4f} {rev_rate:>12}")

    # ================================================================
    # 分析7: 低覆盖/低SNR警告
    # ================================================================
    w("\n" + "=" * 70)
    w("分析7: 低覆盖/低SNR警告")
    w("=" * 70)

    # 低覆盖率概念
    low_coverage = []
    for ci in range(n_concepts):
        n_covered = sum(1 for item in data if any(p > 0.01 for p in item["level_probs"][ci]))
        rate = n_covered / n_samples * 100
        if rate < 99.0:
            low_coverage.append((concept_names[ci], concept_types[ci], rate))

    if low_coverage:
        w(f"\n  低覆盖率概念 (<99%):")
        for name, ctype, rate in sorted(low_coverage, key=lambda x: x[2]):
            w(f"    {name} ({ctype}): {rate:.1f}%")
    else:
        w(f"\n  所有概念Verbalizer覆盖率 >= 99%，无异常。")

    # 低SNR概念
    low_snr = [(name, ct, snr, direction) for name, ct, snr, direction in snr_list if snr < 0.1]
    if low_snr:
        w(f"\n  低SNR概念 (<0.1):")
        for name, ctype, snr, direction in low_snr:
            w(f"    {name} ({ctype}): SNR={snr:.4f} {direction}")
    else:
        w(f"\n  所有概念SNR >= 0.1，无异常。")

    # 保存报告
    output_dir = project_root / "experiments"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"verbalizer_coverage_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    w(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
