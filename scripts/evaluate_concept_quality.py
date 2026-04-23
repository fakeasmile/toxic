"""形容词概念向量质量检测脚本

评估生成的形容词概念向量在以下维度的质量：
1. 分布统计：整体均值、标准差、值域、分位数
2. 区分度：Toxic vs Non-Toxic 的均值差异、效应量、逐维度t检验
3. 相关性：各形容词与标签的Pearson/Spearman相关
4. 稀疏性：低值/高值占比，信息集中度
5. 基线偏移：Non-Toxic样本均值偏离0.5的程度
6. 模型间一致性：两个模型概念向量的逐样本相关性（需指定两组路径）

使用示例：
python scripts/evaluate_concept_quality.py --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template binary
python scripts/evaluate_concept_quality.py --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template likert
python scripts/evaluate_concept_quality.py --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template binary --compare_model_name Qwen2.5-3B-Instruct
"""
import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    parser = argparse.ArgumentParser(description="形容词概念向量质量检测")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称(TOXICN/COLD)")
    parser.add_argument("--model_name", type=str, required=True, help="LLM模型名称")
    parser.add_argument("--template", type=str, default="binary", help="提示词模板类型")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="使用训练集还是测试集")
    parser.add_argument("--compare_model_name", type=str, default=None, help="对比模型的名称（用于模型间一致性分析）")
    return parser.parse_args()


def load_data(config, args):
    """加载概念向量和标签"""
    # 加载概念向量
    concept_dir = config.processed_path / args.dataset_name / args.model_name / args.template
    concept_path = concept_dir / f"concept_{args.mode}.json"
    if not concept_path.exists():
        raise FileNotFoundError(f"概念向量文件未找到: {concept_path}")

    with open(concept_path, "r", encoding="utf-8") as f:
        concept_data = json.load(f)

    concepts = np.array([item["concept"] for item in concept_data])
    labels = np.array([item["toxic"] for item in concept_data])
    print(f"加载完成：{len(concepts)}条样本，概念向量维度={concepts.shape[1]}")
    return concepts, labels, concept_path


def load_adjectives(config):
    """加载形容词词典"""
    df = pd.read_csv(config.adjective_path)
    return df["chinese"].tolist()


def evaluate_distribution(concepts):
    """评估1：分布统计"""
    print("\n" + "=" * 60)
    print("1. 分布统计")
    print("=" * 60)
    print(f"  样本数: {concepts.shape[0]}")
    print(f"  维度数: {concepts.shape[1]}")
    print(f"  整体均值: {concepts.mean():.4f}")
    print(f"  整体标准差: {concepts.std():.4f}")
    print(f"  最小值: {concepts.min():.6f}")
    print(f"  最大值: {concepts.max():.6f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(concepts, p):.4f}")


def evaluate_discrimination(concepts, labels, adjectives):
    """评估2：区分度分析"""
    print("\n" + "=" * 60)
    print("2. 区分度分析")
    print("=" * 60)

    toxic = concepts[labels == 1]
    nontoxic = concepts[labels == 0]
    print(f"  Toxic样本: {len(toxic)}, Non-Toxic样本: {len(nontoxic)}")

    # 整体均值差异
    toxic_mean = toxic.mean()
    nontoxic_mean = nontoxic.mean()
    print(f"  Toxic均值: {toxic_mean:.4f}")
    print(f"  Non-Toxic均值: {nontoxic_mean:.4f}")
    print(f"  均值差异: {toxic_mean - nontoxic_mean:.4f}")

    # 逐维度分析
    dim_diffs = toxic.mean(axis=0) - nontoxic.mean(axis=0)
    abs_diffs = np.abs(dim_diffs)

    # 逐维度t检验
    n_dims = concepts.shape[1]
    p_values = np.zeros(n_dims)
    for d in range(n_dims):
        _, p_values[d] = stats.ttest_ind(toxic[:, d], nontoxic[:, d], equal_var=False)
    sig_count = (p_values < 0.05).sum()

    # Cohen's d（逐维度）
    pooled_std = np.sqrt((toxic.var(axis=0) + nontoxic.var(axis=0)) / 2)
    cohens_d = dim_diffs / (pooled_std + 1e-8)

    print(f"  逐维度最大均值差异: {abs_diffs.max():.4f} (形容词: {adjectives[abs_diffs.argmax()]})")
    print(f"  逐维度中位均值差异: {np.median(abs_diffs):.4f}")
    print(f"  显著相关维度数(p<0.05): {sig_count}/{n_dims}")
    print(f"  最大Cohen's d: {np.abs(cohens_d).max():.4f} (形容词: {adjectives[np.abs(cohens_d).argmax()]})")
    print(f"  中位Cohen's d: {np.median(np.abs(cohens_d)):.4f}")

    # Top-10 区分度最高的形容词
    top10_idx = np.argsort(abs_diffs)[::-1][:10]
    print(f"\n  Top-10 区分度最高的形容词:")
    print(f"  {'排名':<4} {'形容词':<12} {'Toxic均值':<10} {'Non-Toxic均值':<14} {'差异':<8} {'Cohen-d':<8} {'p值':<10}")
    for rank, idx in enumerate(top10_idx, 1):
        adj_name = adjectives[idx] if idx < len(adjectives) else f"dim_{idx}"
        print(f"  {rank:<4} {adj_name:<12} {toxic[:, idx].mean():<10.4f} {nontoxic[:, idx].mean():<14.4f} {dim_diffs[idx]:<8.4f} {cohens_d[idx]:<8.4f} {p_values[idx]:<10.2e}")

    return dim_diffs, p_values, cohens_d


def evaluate_correlation(concepts, labels, adjectives):
    """评估3：相关性分析"""
    print("\n" + "=" * 60)
    print("3. 相关性分析")
    print("=" * 60)

    n_dims = concepts.shape[1]
    pearson_r = np.zeros(n_dims)
    spearman_r = np.zeros(n_dims)
    for d in range(n_dims):
        pearson_r[d], _ = stats.pearsonr(concepts[:, d], labels)
        spearman_r[d], _ = stats.spearmanr(concepts[:, d], labels)

    abs_pearson = np.abs(pearson_r)
    print(f"  平均|Pearson r|: {abs_pearson.mean():.4f}")
    print(f"  最大|Pearson r|: {abs_pearson.max():.4f} (形容词: {adjectives[abs_pearson.argmax()]})")
    print(f"  中位|Pearson r|: {np.median(abs_pearson):.4f}")
    print(f"  平均|Spearman r|: {np.abs(spearman_r).mean():.4f}")
    print(f"  最大|Spearman r|: {np.abs(spearman_r).max():.4f} (形容词: {adjectives[np.abs(spearman_r).argmax()]})")

    # Top-10 相关性最强的形容词
    top10_idx = np.argsort(abs_pearson)[::-1][:10]
    print(f"\n  Top-10 Pearson相关最强的形容词:")
    print(f"  {'排名':<4} {'形容词':<12} {'Pearson r':<10} {'Spearman r':<12}")
    for rank, idx in enumerate(top10_idx, 1):
        adj_name = adjectives[idx] if idx < len(adjectives) else f"dim_{idx}"
        print(f"  {rank:<4} {adj_name:<12} {pearson_r[idx]:<10.4f} {spearman_r[idx]:<12.4f}")


def evaluate_sparsity(concepts):
    """评估4：稀疏性分析"""
    print("\n" + "=" * 60)
    print("4. 稀疏性分析")
    print("=" * 60)

    low_ratio = (concepts < 0.1).mean()
    mid_ratio = ((concepts >= 0.4) & (concepts <= 0.6)).mean()
    high_ratio = (concepts > 0.9).mean()
    extreme_ratio = (concepts > 0.95).mean()

    print(f"  <0.1 占比: {low_ratio:.4f}")
    print(f"  0.4~0.6 占比: {mid_ratio:.4f}")
    print(f"  >0.9 占比: {high_ratio:.4f}")
    print(f"  >0.95 占比: {extreme_ratio:.4f}")

    # 信息熵：逐样本计算概念向量的信息熵，衡量信息集中度
    # 将概念向量视为概率分布（先clip到[eps,1]再归一化）
    eps = 1e-8
    clipped = np.clip(concepts, eps, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = clipped / row_sums
    entropy = -np.sum(normalized * np.log(normalized + eps), axis=1)
    max_entropy = np.log(concepts.shape[1])
    print(f"  平均信息熵: {entropy.mean():.4f} (最大可能熵: {max_entropy:.4f})")
    print(f"  归一化熵: {entropy.mean() / max_entropy:.4f}")


def evaluate_baseline_shift(concepts, labels):
    """评估5：基线偏移分析"""
    print("\n" + "=" * 60)
    print("5. 基线偏移分析")
    print("=" * 60)

    nontoxic = concepts[labels == 0]
    nontoxic_mean = nontoxic.mean()

    print(f"  Non-Toxic样本整体均值: {nontoxic_mean:.4f}")
    print(f"  偏离0.5的程度: {abs(nontoxic_mean - 0.5):.4f}")
    if nontoxic_mean > 0.6:
        print(f"  ⚠ 肯定偏误：Non-Toxic均值>{0.6}，模型倾向对中性文本给出高分")
    elif nontoxic_mean < 0.4:
        print(f"  ⚠ 否定偏误：Non-Toxic均值<{0.4}，模型倾向对中性文本给出低分")
    else:
        print(f"  ✓ 基线偏移在可接受范围内")

    # 逐维度基线偏移
    dim_nontoxic_mean = nontoxic.mean(axis=0)
    dim_shift = np.abs(dim_nontoxic_mean - 0.5)
    print(f"  逐维度基线偏移(偏离0.5)均值: {dim_shift.mean():.4f}")
    print(f"  逐维度基线偏移(偏离0.5)中位数: {np.median(dim_shift):.4f}")


def evaluate_model_agreement(config, args, concepts_a):
    """评估6：模型间一致性分析"""
    print("\n" + "=" * 60)
    print("6. 模型间一致性分析")
    print("=" * 60)

    # 查找模型A的概念向量文件路径
    concept_dir_a = config.processed_path / args.dataset_name / args.model_name / args.template
    concept_path_a = concept_dir_a / f"concept_{args.mode}.json"
    if not concept_path_a.exists():
        concept_path_a = config.processed_path / f"{args.mode}_with_concepts({args.dataset_name})({args.model_name}).json"
    if not concept_path_a.exists():
        print(f"  模型A概念向量文件未找到")
        return

    # 查找模型B的概念向量文件路径
    compare_name = args.compare_model_name
    concept_dir_b = config.processed_path / args.dataset_name / compare_name / args.template
    concept_path_b = concept_dir_b / f"concept_{args.mode}.json"
    if not concept_path_b.exists():
        concept_path_b = config.processed_path / f"{args.mode}_with_concepts({args.dataset_name})({compare_name}).json"
    if not concept_path_b.exists():
        print(f"  对比模型概念向量文件未找到")
        return

    with open(concept_path_a, "r", encoding="utf-8") as f:
        concept_data_a = json.load(f)
    with open(concept_path_b, "r", encoding="utf-8") as f:
        concept_data_b = json.load(f)

    # 按content对齐两个模型的数据
    content_to_concept_b = {item["content"]: item["concept"] for item in concept_data_b}

    concepts_a_list = []
    concepts_b_list = []
    for item in concept_data_a:
        content = item["content"]
        if content in content_to_concept_b:
            concepts_a_list.append(item["concept"])
            concepts_b_list.append(content_to_concept_b[content])

    if len(concepts_a_list) == 0:
        print("  两个模型的概念向量无法按content对齐")
        return

    ca = np.array(concepts_a_list)
    cb = np.array(concepts_b_list)
    print(f"  对齐样本数: {len(ca)}")

    # 逐样本Pearson相关
    sample_corrs = []
    for i in range(len(ca)):
        r, _ = stats.pearsonr(ca[i], cb[i])
        sample_corrs.append(r)
    sample_corrs = np.array(sample_corrs)

    print(f"  逐样本Pearson相关 - 均值: {sample_corrs.mean():.4f}, 中位数: {np.median(sample_corrs):.4f}, 标准差: {sample_corrs.std():.4f}")
    print(f"  逐样本Pearson相关 - 最小值: {sample_corrs.min():.4f}, 最大值: {sample_corrs.max():.4f}")

    # 整体矩阵相关（展平后）
    flat_r, _ = stats.pearsonr(ca.flatten(), cb.flatten())
    print(f"  整体展平Pearson相关: {flat_r:.4f}")

    # 逐维度相关
    dim_corrs = []
    for d in range(ca.shape[1]):
        r, _ = stats.pearsonr(ca[:, d], cb[:, d])
        dim_corrs.append(r)
    dim_corrs = np.array(dim_corrs)
    print(f"  逐维度Pearson相关 - 均值: {dim_corrs.mean():.4f}, 中位数: {np.median(dim_corrs):.4f}")


def main():
    args = parse_args()
    config = MLPConfig()

    # 加载数据
    concepts, labels, concept_path = load_data(config, args)
    adjectives = load_adjectives(config)

    print(f"\n概念向量文件: {concept_path}")
    print(f"数据集: {args.dataset_name}, 模型: {args.model_name}, 模板: {args.template}, 模式: {args.mode}")

    # 执行各项评估
    evaluate_distribution(concepts)
    evaluate_discrimination(concepts, labels, adjectives)
    evaluate_correlation(concepts, labels, adjectives)
    evaluate_sparsity(concepts)
    evaluate_baseline_shift(concepts, labels)

    if args.compare_model_name:
        evaluate_model_agreement(config, args, concepts)

    print("\n" + "=" * 60)
    print("质量检测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
