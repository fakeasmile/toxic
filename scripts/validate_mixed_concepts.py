"""验证混合概念向量的信号质量。

在小样本（200条）上快速验证：
  1. 各类型概念的LR分类F1（单独 vs 组合）
  2. 新概念与现有形容词概念的相关性（是否提供独立信息）
  3. 概念级SNR分析
  4. 对FP/FN样本的修正潜力

使用示例：
    python scripts/validate_mixed_concepts.py --dataset_name TOXICN --model_name glm-4-9b-chat
    python scripts/validate_mixed_concepts.py --dataset_name TOXICN --model_name glm-4-9b-chat --num_samples 200
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    parser = argparse.ArgumentParser(description="验证混合概念向量信号质量")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--concept_csv', type=str, default='mixed_concepts_v3.csv')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='仅使用前N个样本（默认全部）')
    parser.add_argument('--compare_adjective', action='store_true',
                        help='同时加载现有形容词概念向量进行对比')
    parser.add_argument('--adjective_name', type=str, default='toxic_adjectives_v2.csv')
    parser.add_argument('--adjective_suffix', type=str, default='3level')
    return parser.parse_args()


def load_mixed_concepts(concept_path, num_samples=None):
    """加载混合概念向量，按类型提取特征。"""
    with open(concept_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if num_samples:
        raw = raw[:num_samples]

    # 获取概念信息
    concept_names = raw[0]["concept_names"]
    concept_types = raw[0]["concept_types"]

    labels = np.array([item["toxic"] for item in raw])

    # 提取各类特征
    features_by_type = {}
    for ctype in ["intent", "mode", "target", "adjective"]:
        indices = [i for i, t in enumerate(concept_types) if t == ctype]
        if not indices:
            continue
        # 对二元概念: 用P(是)=level_probs[-1]; 对3级概念: 用P(1)和P(3)
        feat_list = []
        for item in raw:
            row = []
            for idx in indices:
                probs = item["level_probs"][idx]
                if ctype in ("intent", "mode", "target"):
                    # 二元: P(否), P(是)
                    row.extend([probs[0], probs[1]])  # P(1=否), P(2=是)
                else:
                    # 3级: P(1), P(3) (P(2)是噪声)
                    row.extend([probs[0], probs[2]])
            feat_list.append(row)
        features_by_type[ctype] = (np.array(feat_list), indices)

    # 全部特征
    all_feat = []
    for item in raw:
        row = []
        for i, (ctype, probs) in enumerate(zip(concept_types, item["level_probs"])):
            if ctype in ("intent", "mode", "target"):
                row.extend([probs[0], probs[1]])
            else:
                row.extend([probs[0], probs[2]])
        all_feat.append(row)
    all_features = np.array(all_feat)

    return all_features, features_by_type, labels, concept_names, concept_types


def load_adjective_concepts(concept_path, num_samples=None):
    """加载现有3级形容词概念向量。"""
    with open(concept_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if num_samples:
        raw = raw[:num_samples]

    labels = np.array([item["toxic"] for item in raw])
    feat_list = []
    for item in raw:
        row = []
        for probs in item["level_probs"]:
            row.extend([probs[0], probs[2]])  # P(1), P(3)
        feat_list.append(row)

    return np.array(feat_list), labels


def lr_cv_f1(X, y, n_splits=5):
    """5折交叉验证LR F1。"""
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    return scores.mean(), scores.std()


def compute_concept_snr(features, labels, concept_names, concept_types):
    """计算每个概念的SNR。"""
    snr_list = []
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0

    for i, (name, ctype) in enumerate(zip(concept_names, concept_types)):
        # 取主信号: 二元→P(是)=feat[:, 2*i+1], 3级→P(3)=feat[:, 2*i+1]
        sig = features[:, 2 * i + 1]

        # 信号强度: 有毒/无毒均值差
        mean_toxic = sig[toxic_mask].mean()
        mean_nontoxic = sig[nontoxic_mask].mean()
        signal = abs(mean_toxic - mean_nontoxic)

        # 噪声: 组内标准差
        noise = sig[toxic_mask].std() + sig[nontoxic_mask].std() + 1e-8

        snr = signal / noise
        snr_list.append({
            'name': name, 'type': ctype,
            'mean_toxic': mean_toxic, 'mean_nontoxic': mean_nontoxic,
            'signal': signal, 'noise': noise, 'snr': snr
        })

    return sorted(snr_list, key=lambda x: x['snr'], reverse=True)


def compute_cross_correlation(feat_a, names_a, feat_b, names_b, top_k=10):
    """计算两组概念特征之间的交叉相关性。"""
    # 取主信号 (P(是) or P(3))
    sig_a = feat_a[:, 1::2]  # 每个概念的第2个特征
    sig_b = feat_b[:, 1::2]

    corr_matrix = np.corrcoef(sig_a.T, sig_b.T)
    n_a = len(names_a)

    # 取跨组相关性 (a vs b)
    cross_corr = corr_matrix[:n_a, n_a:]

    # 找最高相关的概念对
    pairs = []
    for i in range(n_a):
        for j in range(len(names_b)):
            pairs.append({
                'concept_a': names_a[i], 'concept_b': names_b[j],
                'r': cross_corr[i, j]
            })
    pairs.sort(key=lambda x: abs(x['r']), reverse=True)

    return pairs[:top_k], cross_corr


def main():
    args = parse_args()
    config = MLPConfig()

    concept_stem = Path(args.concept_csv).stem
    mixed_path = (config.processed_path / args.dataset_name / args.model_name /
                  f"concept_train_{args.model_name}_{concept_stem}_mixed.json")

    if not mixed_path.exists():
        raise FileNotFoundError(f"混合概念向量不存在: {mixed_path}")

    # 加载混合概念
    print("加载混合概念向量...")
    all_features, features_by_type, labels, concept_names, concept_types = load_mixed_concepts(
        mixed_path, args.num_samples
    )

    print(f"样本数: {len(labels)} (有毒={labels.sum()}, 无毒={len(labels)-labels.sum()})")
    print(f"总特征维度: {all_features.shape[1]}")
    for ctype, (feat, indices) in features_by_type.items():
        print(f"  {ctype}: {len(indices)}概念 × 2维 = {feat.shape[1]}维")

    # ============================================================
    # 分析1: 各类型概念的分类F1
    # ============================================================
    print("\n" + "=" * 60)
    print("分析1: 各类型概念的LR分类F1 (5折CV)")
    print("=" * 60)

    results = {}
    for ctype, (feat, indices) in features_by_type.items():
        f1_mean, f1_std = lr_cv_f1(feat, labels)
        results[ctype] = (f1_mean, f1_std)
        print(f"  仅{ctype} ({len(indices)}概念, {feat.shape[1]}维): F1={f1_mean:.4f} ± {f1_std:.4f}")

    # 全部特征
    f1_all, std_all = lr_cv_f1(all_features, labels)
    results['all_mixed'] = (f1_all, std_all)
    print(f"  全部混合概念 ({all_features.shape[1]}维): F1={f1_all:.4f} ± {std_all:.4f}")

    # 各类型两两组合
    type_list = list(features_by_type.keys())
    for i in range(len(type_list)):
        for j in range(i + 1, len(type_list)):
            t1, t2 = type_list[i], type_list[j]
            combined = np.hstack([features_by_type[t1][0], features_by_type[t2][0]])
            f1_comb, std_comb = lr_cv_f1(combined, labels)
            results[f'{t1}+{t2}'] = (f1_comb, std_comb)
            print(f"  {t1}+{t2} ({combined.shape[1]}维): F1={f1_comb:.4f} ± {std_comb:.4f}")

    # ============================================================
    # 分析2: 与现有形容词概念的对比
    # ============================================================
    if args.compare_adjective:
        print("\n" + "=" * 60)
        print("分析2: 与现有形容词概念的对比")
        print("=" * 60)

        adj_stem = Path(args.adjective_name).stem
        suffix = adj_stem.replace("toxic_adjectives_", "")
        adj_path = (config.processed_path / args.dataset_name / args.model_name /
                    f"concept_train_{args.model_name}_{suffix}_{args.adjective_suffix}.json")

        if not adj_path.exists():
            print(f"  形容词概念向量不存在: {adj_path}，跳过对比")
        else:
            adj_features, adj_labels = load_adjective_concepts(adj_path, args.num_samples)

            # 确保样本对齐
            min_n = min(len(labels), len(adj_labels))
            adj_features = adj_features[:min_n]
            adj_labels_sub = adj_labels[:min_n]
            mixed_feat_sub = all_features[:min_n]
            labels_sub = labels[:min_n]

            # 形容词F1
            f1_adj, std_adj = lr_cv_f1(adj_features, labels_sub)
            print(f"  现有形容词 ({adj_features.shape[1]}维): F1={f1_adj:.4f} ± {std_adj:.4f}")

            # 组合
            combined_all = np.hstack([adj_features, mixed_feat_sub])
            f1_combined, std_combined = lr_cv_f1(combined_all, labels_sub)
            print(f"  形容词+混合概念 ({combined_all.shape[1]}维): F1={f1_combined:.4f} ± {std_combined:.4f}")

            # 交叉相关性
            adj_names = []
            import csv
            with open(config.raw_data_path / "adjective" / args.adjective_name, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    adj_names.append(row[1] if len(row) > 1 else row[0])

            # 新概念主信号
            new_sig_names = concept_names
            new_sig_feat = all_features[:min_n, 1::2]

            top_pairs, cross_corr = compute_cross_correlation(
                adj_features, adj_names, new_sig_feat, new_sig_names, top_k=10
            )

            print(f"\n  新概念与形容词最高相关的Top-10概念对:")
            for p in top_pairs:
                print(f"    {p['concept_a']} ↔ {p['concept_b']}: r={p['r']:.4f}")

            # 新概念对形容词的独立性
            max_corr_per_new = np.abs(cross_corr).max(axis=0)
            print(f"\n  新概念独立性 (与形容词最高|r|的均值): {max_corr_per_new.mean():.4f}")
            for i, name in enumerate(new_sig_names):
                print(f"    {name}: max|r|={max_corr_per_new[i]:.4f}")

    # ============================================================
    # 分析3: 概念SNR
    # ============================================================
    print("\n" + "=" * 60)
    print("分析3: 概念SNR排名")
    print("=" * 60)

    snr_list = compute_concept_snr(all_features, labels, concept_names, concept_types)
    for s in snr_list:
        print(f"  {s['name']} ({s['type']}): SNR={s['snr']:.4f}, "
              f"有毒均值={s['mean_toxic']:.4f}, 无毒均值={s['mean_nontoxic']:.4f}")

    # 按类型汇总
    for ctype in ["intent", "mode", "target", "adjective"]:
        type_snrs = [s['snr'] for s in snr_list if s['type'] == ctype]
        if type_snrs:
            print(f"\n  {ctype}类SNR: 均值={np.mean(type_snrs):.4f}, "
                  f"最高={np.max(type_snrs):.4f}, 最低={np.min(type_snrs):.4f}")

    # ============================================================
    # 分析4: FP/FN修正潜力
    # ============================================================
    print("\n" + "=" * 60)
    print("分析4: 新概念对FP/FN的修正潜力")
    print("=" * 60)

    # 用形容词概念训练LR，找误分类样本
    if 'adjective' in features_by_type:
        adj_feat, adj_indices = features_by_type['adjective']
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(adj_feat, labels)
        adj_preds = clf.predict(adj_feat)

        fn_mask = (adj_preds == 0) & (labels == 1)  # 有毒→判无毒
        fp_mask = (adj_preds == 1) & (labels == 0)  # 无毒→判有毒

        print(f"形容词概念误分类: FN={fn_mask.sum()}, FP={fp_mask.sum()}")

        # 检查新概念在误分类样本上的信号
        for ctype in ["intent", "mode", "target"]:
            if ctype not in features_by_type:
                continue
            new_feat, _ = features_by_type[ctype]
            # 取主信号
            sig_indices = list(range(1, new_feat.shape[1], 2))  # P(是)
            for idx, sig_idx in enumerate(sig_indices):
                sig = new_feat[:, sig_idx]
                name = concept_names[[i for i, t in enumerate(concept_types) if t == ctype][idx]]
                fn_mean = sig[fn_mask].mean() if fn_mask.sum() > 0 else 0
                fp_mean = sig[fp_mask].mean() if fp_mask.sum() > 0 else 0
                tp_mean = sig[(adj_preds == 1) & (labels == 1)].mean()
                tn_mean = sig[(adj_preds == 0) & (labels == 0)].mean()
                print(f"  {name}: FN均值={fn_mean:.4f}, FP均值={fp_mean:.4f}, "
                      f"TP均值={tp_mean:.4f}, TN均值={tn_mean:.4f}")

    # ============================================================
    # 保存报告
    # ============================================================
    report_path = Path(r"d:\toxicnew\experiments") / f"mixed_concept_validation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("混合概念向量验证报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 各类型概念LR分类F1 (5折CV)\n")
        f.write("-" * 40 + "\n")
        for k, (v_mean, v_std) in sorted(results.items()):
            f.write(f"  {k}: F1={v_mean:.4f} ± {v_std:.4f}\n")

        f.write(f"\n2. 概念SNR排名\n")
        f.write("-" * 40 + "\n")
        for s in snr_list:
            f.write(f"  {s['name']} ({s['type']}): SNR={s['snr']:.4f}, "
                    f"有毒={s['mean_toxic']:.4f}, 无毒={s['mean_nontoxic']:.4f}\n")

        f.write(f"\n3. 各类型SNR汇总\n")
        f.write("-" * 40 + "\n")
        for ctype in ["intent", "mode", "target", "adjective"]:
            type_snrs = [s['snr'] for s in snr_list if s['type'] == ctype]
            if type_snrs:
                f.write(f"  {ctype}: 均值={np.mean(type_snrs):.4f}, "
                        f"最高={np.max(type_snrs):.4f}, 最低={np.min(type_snrs):.4f}\n")

    print(f"\n报告已保存至: {report_path}")


if __name__ == '__main__':
    main()
