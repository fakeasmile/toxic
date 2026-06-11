"""语用推理结果诊断工具

分析 generate_pragmatic_reasoning.py 的输出，统计各维度失败率，
输出典型失败样本，帮助定位问题。

使用示例：
python scripts/diagnose_reasoning.py --input_path data/processed/TOXICN/Qwen2.5-7B-Instruct/pragmatic_reasoning_train.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.generate_pragmatic_reasoning import DIMENSION_CONFIG, DIMENSION_NAMES


def main():
    parser = argparse.ArgumentParser(description="语用推理结果诊断")
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--show_n', type=int, default=5, help='显示N个失败样本')
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"总样本数: {total}\n")

    # 统计各维度失败率
    dim_fail_counts = {dim: 0 for dim in DIMENSION_NAMES}
    dim_concept_counter = {dim: Counter() for dim in DIMENSION_NAMES}
    full_fail = 0  # 7个维度全失败
    partial_fail = 0  # 部分维度失败
    full_success = 0

    for item in data:
        reasoning = item["reasoning"]
        fail_dims = 0
        for dim in DIMENSION_NAMES:
            concept = reasoning[dim]["concept"]
            dim_concept_counter[dim][concept] += 1
            if concept == "解析失败":
                dim_fail_counts[dim] += 1
                fail_dims += 1

        if fail_dims == 7:
            full_fail += 1
        elif fail_dims > 0:
            partial_fail += 1
        else:
            full_success += 1

    print("=" * 60)
    print("整体统计")
    print("=" * 60)
    print(f"完全成功(7/7维度命中): {full_success} ({full_success/total*100:.1f}%)")
    print(f"部分失败(有维度未命中): {partial_fail} ({partial_fail/total*100:.1f}%)")
    print(f"完全失败(0/7维度命中): {full_fail} ({full_fail/total*100:.1f}%)")
    print(f"至少1个维度失败: {partial_fail + full_fail} ({(partial_fail+full_fail)/total*100:.1f}%)")

    print("\n" + "=" * 60)
    print("各维度失败率")
    print("=" * 60)
    for dim in DIMENSION_NAMES:
        fail_count = dim_fail_counts[dim]
        fail_rate = fail_count / total * 100
        label = DIMENSION_CONFIG[dim]["label"]
        print(f"  {label}({dim}): {fail_count}/{total} = {fail_rate:.1f}%")

    print("\n" + "=" * 60)
    print("各维度概念分布 (Top-5 + 失败)")
    print("=" * 60)
    for dim in DIMENSION_NAMES:
        label = DIMENSION_CONFIG[dim]["label"]
        counter = dim_concept_counter[dim]
        print(f"\n  {label}({dim}):")
        top5 = counter.most_common(5)
        for concept, count in top5:
            pct = count / total * 100
            marker = " ✗" if concept == "解析失败" else " ✓"
            print(f"    {concept}: {count} ({pct:.1f}%){marker}")

    # 显示部分失败样本
    if args.show_n > 0:
        print("\n" + "=" * 60)
        print(f"部分失败样本 (前{args.show_n}个)")
        print("=" * 60)
        shown = 0
        for item in data:
            reasoning = item["reasoning"]
            fail_dims = [dim for dim in DIMENSION_NAMES if reasoning[dim]["concept"] == "解析失败"]
            if 0 < len(fail_dims) < 7:
                print(f"\n--- 样本 (失败维度: {[DIMENSION_CONFIG[d]['label'] for d in fail_dims]}) ---")
                print(f"文本: {item['content'][:80]}")
                print(f"标签: {'有毒' if item['toxic'] else '无毒'}")
                for dim in fail_dims:
                    concept = reasoning[dim]["concept"]
                    reason = reasoning[dim]["reason"]
                    print(f"  {DIMENSION_CONFIG[dim]['label']}: concept='{concept}', reason='{reason[:60]}'")
                shown += 1
                if shown >= args.show_n:
                    break


if __name__ == '__main__':
    main()
