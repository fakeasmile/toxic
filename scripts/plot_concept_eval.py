"""绘制概念评估验证图（对标SCBM Fig.3架构）

输出：D:/papper/figures/concept_eval_p3.pdf
数据源：ToxiCN测试集 level_probs（原始LLM输出，非训练后门控层）
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ====== 路径 ======
PROJECT = Path(__file__).parent.parent if "__file__" in dir() else Path("D:/toxicnew")
DATA_DIR = PROJECT / "data" / "processed" / "TOXICN" / "glm-4-9b-chat"
ADJ_TYPE_PATH = PROJECT / "data" / "raw" / "adjective" / "toxic_adjectives_v2_types.json"
SIMHEI_PATH = PROJECT / "SimHei.ttf"
OUT_DIR = Path("D:/papper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 顶刊审美配置 ======
# 注册 SimHei 中文字体
import matplotlib.font_manager as fm
fm.fontManager.addfont(str(SIMHEI_PATH))
fm.FontProperties(fname=str(SIMHEI_PATH)).get_name()  # 触发字体扫描

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ====== 加载数据 ======
with open(DATA_DIR / "concept_test_glm-4-9b-chat_v2_3level.json", encoding="utf-8") as f:
    test_data = json.load(f)
with open(ADJ_TYPE_PATH, encoding="utf-8") as f:
    type_list = [item["type"] for item in json.load(f)]

n_samples = len(test_data)
n_concepts = len(test_data[0]["concept"])
print(f"测试集: {n_samples} 样本, {n_concepts} 概念")

# 提取p3/p2
p3 = np.zeros((n_samples, n_concepts))
p2 = np.zeros((n_samples, n_concepts))
labels = np.zeros(n_samples, dtype=int)
for si, item in enumerate(test_data):
    lp = item["level_probs"]
    for ci in range(n_concepts):
        p3[si, ci] = lp[ci][2]
        p2[si, ci] = lp[ci][1]
    labels[si] = item["toxic"]

n_toxic = np.sum(labels == 1)
n_nt = np.sum(labels == 0)
print(f"有毒: {n_toxic}, 无毒: {n_nt}")

# ====== 按类型分组排序 ======
type_order = ["行为类", "歧视类", "评价类", "策略类", "中性类"]
type_names_en = ["behavior", "discrimination", "evaluation", "strategy", "neutral"]
cn_to_idx = {}
for i, t in enumerate(type_names_en):
    cn_to_idx[t] = type_order[i]

# 为每个概念分配中文类型名
adj_types_cn = [cn_to_idx[t] for t in type_list]

# 按类型分组排序
sorted_indices = []
type_boundaries = []  # 记录类型分界位置
for tp in type_order:
    group = [i for i, t in enumerate(adj_types_cn) if t == tp]
    sorted_indices.extend(group)
    if group:
        type_boundaries.append((tp, len(sorted_indices)))  # 终点索引

# 重新排列
p3_sorted = p3[:, sorted_indices]
p2_sorted = p2[:, sorted_indices]

# ====== 按label分组计算统计量 ======
toxic_mask = labels == 1
nt_mask = labels == 0

toxic_p3_mean = p3_sorted[toxic_mask].mean(axis=0)
toxic_p3_std = p3_sorted[toxic_mask].std(axis=0)
nt_p3_mean = p3_sorted[nt_mask].mean(axis=0)
nt_p3_std = p3_sorted[nt_mask].std(axis=0)

# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(14, 4.5))

x = np.arange(n_concepts)

# 填充±1σ阴影
ax.fill_between(x, toxic_p3_mean - toxic_p3_std, toxic_p3_mean + toxic_p3_std,
                color="#D32F2F", alpha=0.15, edgecolor="none")
ax.fill_between(x, nt_p3_mean - nt_p3_std, nt_p3_mean + nt_p3_std,
                color="#1976D2", alpha=0.15, edgecolor="none")

# 均值线
ax.plot(x, toxic_p3_mean, color="#D32F2F", linewidth=1.4, label="有毒样本 (均值±1σ)", zorder=3)
ax.plot(x, nt_p3_mean, color="#1976D2", linewidth=1.4, label="无毒样本 (均值±1σ)", zorder=3)

# 类型分隔线 + 标注
prev_boundary = 0
for tp, boundary in type_boundaries:
    if boundary < n_concepts:  # 不是最后一组
        ax.axvline(x=boundary - 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    mid = (prev_boundary + boundary) / 2
    ax.text(mid, ax.get_ylim()[1] * 1.02, tp, ha="center", va="bottom", fontsize=8,
            fontproperties=fm.FontProperties(fname=str(SIMHEI_PATH), size=8))
    prev_boundary = boundary

ax.set_xlabel("形容词（按语义类型分组排列）", fontsize=10)
ax.set_ylabel("平均实施概率 $p_3$", fontsize=10)
ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray", framealpha=0.8)
ax.set_xlim(-1, n_concepts)
ax.set_ylim(bottom=0)

plt.tight_layout()
out_path = OUT_DIR / "concept_eval_p3.pdf"
fig.savefig(out_path, format="pdf")
plt.close(fig)
print(f">>> 已保存: {out_path}")

# ====== 打印 Top 差异形容词（供 Table 用） ======
# 读取形容词中文名
with open(ADJ_TYPE_PATH, encoding="utf-8") as f:
    adj_list = json.load(f)
adj_names_cn = [item["chinese"] for item in adj_list]
adj_names_sorted = [adj_names_cn[idx] for idx in sorted_indices]

diffs = toxic_p3_mean - nt_p3_mean
top_k = min(15, n_concepts)
top_idx = np.argsort(diffs)[::-1][:top_k]
print(f"\nTop-{top_k} 实施概率差异最大的形容词 (Table):")
for i, idx in enumerate(top_idx):
    orig_idx = sorted_indices[idx]
    print(f"  {i+1}. {adj_names_sorted[idx]:<10s}  类型={adj_types_cn[orig_idx]:<6s}  "
          f"有毒均值={toxic_p3_mean[idx]:.4f}  无毒均值={nt_p3_mean[idx]:.4f}  diff={diffs[idx]:.4f}")
