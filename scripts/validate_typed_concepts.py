"""验证类型感知概念向量的信号质量。

评估维度：
1. 各类型概念的SNR（有毒/无毒差值的信噪比）
2. 二元verbalizer vs 3级verbalizer的信号质量对比
3. 新增概念（羞辱意图、冒犯效果等）的信号质量
4. 小规模LR分类F1

使用示例：
    python scripts/validate_typed_concepts.py --concept_file data/processed/TOXICN/glm-4-9b-chat/concept_train_glm-4-9b-chat_v4_mini_typed.json
"""

import argparse
import json
import sys
import io
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="验证类型感知概念向量")
    parser.add_argument("--concept_file", type=str, required=True, help="概念向量JSON文件路径")
    return parser.parse_args()


def compute_snr(toxic_vals, nontoxic_vals):
    """计算信噪比：|差值| / (标准差之和)。"""
    if len(toxic_vals) < 2 or len(nontoxic_vals) < 2:
        return 0.0
    diff = abs(np.mean(toxic_vals) - np.mean(nontoxic_vals))
    std_sum = np.std(toxic_vals) + np.std(nontoxic_vals)
    if std_sum < 1e-8:
        return 0.0
    return diff / std_sum


def lr_cv_f1(X, y, n_splits=3):
    """交叉验证LR F1。"""
    if len(set(y)) < 2 or min(Counter(y).values()) < n_splits:
        return 0.0, 0.0
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    return scores.mean(), scores.std()


class TeeWriter:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, text):
        print(text, end="")
        self.buf.write(text)

    def getvalue(self):
        return self.buf.getvalue()


def main():
    args = parse_args()
    report = TeeWriter()

    # 加载概念向量
    with open(args.concept_file, "r", encoding="utf-8") as f:
        save_data = json.load(f)

    meta = save_data.get("meta", {})
    data = save_data.get("data", save_data)  # 兼容无meta格式

    concept_names = meta.get("concept_names", [])
    concept_types = meta.get("concept_types", [])

    # 如果meta中没有概念信息，尝试从第一条数据推断
    if not concept_names and data:
        n_concepts = len(data[0].get("concept_scores", []))
        concept_names = [f"concept_{i}" for i in range(n_concepts)]
        concept_types = ["unknown"] * n_concepts

    labels = np.array([item["toxic"] for item in data])
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0

    report.write(f"样本数: {len(labels)} (有毒={toxic_mask.sum()}, 无毒={nontoxic_mask.sum()})\n")
    report.write(f"概念数: {len(concept_names)}\n")
    type_dist = Counter(concept_types)
    for t, c in sorted(type_dist.items()):
        report.write(f"  {t}: {c}概念\n")

    # 3级/二元类型
    THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
    BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}

    # ============================================================
    # 分析1: 各类型概念逐个SNR分析
    # ============================================================
    report.write("\n" + "=" * 60 + "\n")
    report.write("分析1: 逐概念SNR分析\n")
    report.write("=" * 60 + "\n")

    concept_snr = []
    for i, (name, ptype) in enumerate(zip(concept_names, concept_types)):
        if ptype in THREE_LEVEL_TYPES:
            # 3级类型：P(3=有害表达)作为主信号
            toxic_vals = [item["level_probs"][i][2] for item in data if item["toxic"] == 1]
            nontoxic_vals = [item["level_probs"][i][2] for item in data if item["toxic"] == 0]
            signal = "P(3=有害表达)"

            # P(2=存在但无害)也分析
            toxic_p2 = [item["level_probs"][i][1] for item in data if item["toxic"] == 1]
            nontoxic_p2 = [item["level_probs"][i][1] for item in data if item["toxic"] == 0]
            snr_p2 = compute_snr(toxic_p2, nontoxic_p2)
            diff_p2 = np.mean(toxic_p2) - np.mean(nontoxic_p2)
        else:
            # 二元类型：P(2=是)作为主信号
            toxic_vals = [item["level_probs"][i][1] for item in data if item["toxic"] == 1]
            nontoxic_vals = [item["level_probs"][i][1] for item in data if item["toxic"] == 0]
            signal = "P(2=是)"
            snr_p2 = 0
            diff_p2 = 0

        snr = compute_snr(toxic_vals, nontoxic_vals)
        diff = np.mean(toxic_vals) - np.mean(nontoxic_vals)
        direction = "有毒高" if diff > 0 else "无毒高"

        concept_snr.append({
            "name": name, "type": ptype, "snr": snr,
            "diff": diff, "direction": direction, "signal": signal,
            "snr_p2": snr_p2, "diff_p2": diff_p2,
        })

    # 按类型分组输出
    for ptype in ["behavior", "strategy", "evaluation", "discrimination", "neutral", "intent", "effect"]:
        type_concepts = [c for c in concept_snr if c["type"] == ptype]
        if not type_concepts:
            continue
        v_type = "3级危害锚定" if ptype in THREE_LEVEL_TYPES else "二元判断"
        report.write(f"\n  [{ptype}] ({v_type}):\n")
        for c in sorted(type_concepts, key=lambda x: -x["snr"]):
            report.write(f"    {c['name']}: SNR={c['snr']:.4f}, {c['signal']}差值={c['diff']:.4f} ({c['direction']})")
            if ptype in THREE_LEVEL_TYPES:
                p2_dir = "有毒高" if c["diff_p2"] > 0 else "无毒高"
                report.write(f", P(2)SNR={c['snr_p2']:.4f} ({p2_dir})")
            report.write("\n")

    # ============================================================
    # 分析2: 不同类型verbalizer的信号质量对比
    # ============================================================
    report.write("\n" + "=" * 60 + "\n")
    report.write("分析2: 二元 vs 3级 verbalizer信号质量\n")
    report.write("=" * 60 + "\n")

    for ptype in ["behavior", "strategy", "evaluation", "discrimination", "neutral", "intent", "effect"]:
        type_indices = [i for i, t in enumerate(concept_types) if t == ptype]
        if not type_indices:
            continue

        # 收集该类型的所有概率特征
        if ptype in THREE_LEVEL_TYPES:
            # 3级：用P(1)+P(2)+P(3)
            feats = []
            for item in data:
                row = []
                for idx in type_indices:
                    probs = item["level_probs"][idx]
                    row.extend(probs)
                feats.append(row)
        else:
            # 二元：用P(1)+P(2)
            feats = []
            for item in data:
                row = []
                for idx in type_indices:
                    probs = item["level_probs"][idx]
                    row.extend(probs)
                feats.append(row)

        X = np.array(feats)
        f1_mean, f1_std = lr_cv_f1(X, labels)
        avg_snr = np.mean([c["snr"] for c in concept_snr if c["type"] == ptype])
        report.write(f"  {ptype}: {len(type_indices)}概念, 平均SNR={avg_snr:.4f}, LR F1={f1_mean:.4f}±{f1_std:.4f}\n")

    # ============================================================
    # 分析3: 全概念组合分类
    # ============================================================
    report.write("\n" + "=" * 60 + "\n")
    report.write("分析3: 全概念组合LR分类\n")
    report.write("=" * 60 + "\n")

    # 构建完整特征矩阵
    all_feats = []
    for item in data:
        row = []
        for i, ptype in enumerate(concept_types):
            probs = item["level_probs"][i]
            if ptype in THREE_LEVEL_TYPES:
                row.extend(probs)  # P(1), P(2), P(3)
            else:
                row.extend(probs)  # P(1), P(2)
        all_feats.append(row)
    X_all = np.array(all_feats)

    f1_mean, f1_std = lr_cv_f1(X_all, labels)
    report.write(f"  全概念({len(concept_names)}个, {X_all.shape[1]}维): LR F1={f1_mean:.4f}±{f1_std:.4f}\n")

    # 分类型组合
    for group_name, group_types in [
        ("形容词概念(behavior+strategy+evaluation+discrimination+neutral)",
         ["behavior", "strategy", "evaluation", "discrimination", "neutral"]),
        ("语用概念(intent+effect)", ["intent", "effect"]),
        ("二元类型(behavior+strategy+intent+effect)",
         ["behavior", "strategy", "intent", "effect"]),
        ("3级类型(evaluation+discrimination+neutral)",
         ["evaluation", "discrimination", "neutral"]),
    ]:
        group_indices = [i for i, t in enumerate(concept_types) if t in group_types]
        if not group_indices:
            continue
        feats = []
        for item in data:
            row = []
            for idx in group_indices:
                probs = item["level_probs"][idx]
                if concept_types[idx] in THREE_LEVEL_TYPES:
                    row.extend(probs)
                else:
                    row.extend(probs)
            feats.append(row)
        X_group = np.array(feats)
        f1_mean, f1_std = lr_cv_f1(X_group, labels)
        report.write(f"  {group_name}: {len(group_indices)}概念, {X_group.shape[1]}维, LR F1={f1_mean:.4f}±{f1_std:.4f}\n")

    # ============================================================
    # 分析4: 新增/修改概念的信号验证
    # ============================================================
    report.write("\n" + "=" * 60 + "\n")
    report.write("分析4: 新增/修改概念信号验证\n")
    report.write("=" * 60 + "\n")

    new_concepts = [
        "反串黑的", "冒犯但非恶意的", "羞辱意图", "冒犯效果",
        "腐蚀性的", "偏见意图", "外貌与身体歧视的",
        "教育启发效果", "共情凝聚效果",
    ]
    for name in new_concepts:
        if name in concept_names:
            idx = concept_names.index(name)
            c = concept_snr[idx]
            report.write(f"  {name} [{c['type']}]: SNR={c['snr']:.4f}, 差值={c['diff']:.4f} ({c['direction']})")
            if c["type"] in THREE_LEVEL_TYPES:
                p2_dir = "有毒高" if c["diff_p2"] > 0 else "无毒高"
                report.write(f", P(2)SNR={c['snr_p2']:.4f} ({p2_dir})")
            report.write("\n")
        else:
            report.write(f"  {name}: 未在小词典中\n")

    # ============================================================
    # 分析5: P(2)方向验证（anti-pattern信号）
    # ============================================================
    report.write("\n" + "=" * 60 + "\n")
    report.write("分析5: 3级概念P(2=存在但无害)方向验证\n")
    report.write("=" * 60 + "\n")

    p2_reversal = 0
    p2_total = 0
    for c in concept_snr:
        if c["type"] in THREE_LEVEL_TYPES and c["snr_p2"] > 0:
            p2_total += 1
            if c["diff_p2"] < 0:  # 无毒文本P(2)更高 → anti-pattern
                p2_reversal += 1

    if p2_total > 0:
        report.write(f"  P(2)方向反转(无毒高): {p2_reversal}/{p2_total} ({p2_reversal/p2_total*100:.1f}%)\n")
    else:
        report.write(f"  无3级类型概念\n")

    # ============================================================
    # 保存报告
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = project_root / "experiments" / f"typed_concept_validation_{timestamp}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.getvalue())

    report.write(f"\n报告已保存至: {report_path}\n")


if __name__ == "__main__":
    main()
