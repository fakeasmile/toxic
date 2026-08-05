"""v2 3level概念向量分类器 — 基于rel_enact稳健架构+P2反模式增量信号。

策略：保守增量，一次只加一个改动
  - 基底：rel_enact (264维 = relevance(132) + enactment(132))
  - 增量：P2_raw (132维) — 显式"涉及但未实施"反模式信号
  - 门控：TypeAugmentedGated风格（全矩阵门控 + 类型级统计特征注入分类层）
  - 训练：完全标准配置（AdamW/OneCycleLR/batch16/dropout0.5/patience20）

特征维度 = relevance(132) + enactment(132) + P2_raw(132) = 396维
          + 5类型 × 3统计(rel_mean/enact_mean/p2_mean) = 15维
          = 411维 （类型统计仅15维增量，轻量不膨胀）

随机性策略（探索阶段）：
  - 全链路随机，不固定任何seed：split用random_state=None，模型初始化随机，CUDA非确定性
  - 每个trial保存split索引（train_idx/val_idx）供事后复现
  - 论文阶段确认效果后，再用固定seed复现最佳结果

SAM（Sharpness-Aware Minimization）支持：
  --use_sam 开启SAM优化器，--sam_rho 控制扰动半径（默认0.05）
  SAM通过在参数空间的"更差邻居"处计算梯度，引导模型走向平坦极小值，
  降低对split和初始化的敏感性，适合解决val-test脱相关的方差问题。

使用示例：
    # 标准训练（5 trials，全随机）
    python scripts/train_v2_3level_p2signal.py --dataset_name TOXICN --model_name glm-4-9b-chat --n_trials 5
    # SAM训练（5 trials，全随机）
    python scripts/train_v2_3level_p2signal.py --dataset_name TOXICN --model_name glm-4-9b-chat --n_trials 5 --use_sam
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


# =============================================================================
# SAM（Sharpness-Aware Minimization）优化步
# =============================================================================
def sam_step(model, optimizer, criterion, x, y, rho=0.05):
    """SAM单步优化：在参数空间的"更差邻居"处计算梯度，引导平坦极小值。

    流程：
      1. 前向-反向得到原始梯度g
      2. 沿g方向扰动参数到更差点 θ' = θ + ρ·g/||g||
      3. 在θ'处前向-反向得到扰动梯度g'
      4. 恢复原始参数θ，用g'更新（optimizer.step由调用者执行）

    Args:
        model: 待训练的模型
        optimizer: 优化器（调用前需zero_grad）
        criterion: 损失函数
        x, y: 当前batch的输入和标签
        rho: 扰动半径，控制SAM强度

    Returns:
        loss_value: 原始参数处的loss值（用于日志记录）
    """
    # 第一次前向-反向：获取原始梯度g
    loss1 = criterion(model(x), y)
    loss1.backward()

    # 计算梯度L2范数
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # 保存原始参数并扰动（沿梯度上升方向走到更差点）
    original_params = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            original_params[name] = p.data.clone()
            p.data.add_(rho * p.grad / (grad_norm + 1e-8))

    # 第二次前向-反向：在扰动点处获取梯度g'
    optimizer.zero_grad()
    loss2 = criterion(model(x), y)
    loss2.backward()

    # 恢复原始参数（.grad仍保持g'，供optimizer.step使用）
    for name, p in model.named_parameters():
        if name in original_params:
            p.data.copy_(original_params[name])

    return loss1.item()


# =============================================================================
# Mixup 数据增强
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup特征空间插值：x̃ = λx_i + (1-λ)x_j, y为原始索引。

    Args:
        x: (batch, features)
        y: (batch,) long tensor of class indices
        alpha: Beta分布参数（α越大混合越强，0=无mixup）

    Returns:
        mixed_x: 混合后的特征
        y_a, y_b: 两个原始标签索引
        lam: 混合比例λ ~ Beta(α, α)，强制λ ≥ 0.5以避免标签交换
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)  # 确保λ ≥ 0.5，避免标签语义交换

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失：λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b)

    等价于 CE(pred, λ·one_hot(y_a) + (1-λ)·one_hot(y_b))，
    因为交叉熵对目标分布是线性的。
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Concept Orthogonal Loss（COL）— 基于 coop-CBM (NeurIPS 2023)
# =============================================================================
def concept_orthogonal_loss(gated_feats, concept_types, n_concepts):
    """类型级正交损失：鼓励不同概念类型在门控后产生去相关的表示。

    coop-CBM 原论文在概念预测层施加正交约束以提升泛化性。
    适配方案：对 5 种概念类型的门控表示计算跨批次相关系数，
    惩罚非对角平方相关，迫使不同类型捕获不重叠的信号。

    Args:
        gated_feats: (batch, main_dim) 门控后的主特征
        concept_types: 每个概念的类型标签列表
        n_concepts: 概念总数（132 for v2）

    Returns:
        ortho_loss: 标量，非对角平方相关的均值
    """
    batch_size = gated_feats.size(0)
    n_channels = gated_feats.size(1) // n_concepts  # 3 or 4
    type_names = sorted(set(concept_types))
    n_types = len(type_names)  # 5

    # 按类型聚合：每类型所有概念维度取均值 → (batch, n_types)
    type_means = torch.zeros(batch_size, n_types, device=gated_feats.device)
    for t_idx, t in enumerate(type_names):
        indices = [i for i, ct in enumerate(concept_types) if ct == t]
        dims = []
        for ci in indices:
            for ch in range(n_channels):
                dims.append(ci * n_channels + ch)
        type_means[:, t_idx] = gated_feats[:, dims].mean(dim=1)

    # 跨批次去均值 + 归一化 → (batch, n_types)
    type_means = type_means - type_means.mean(dim=0, keepdim=True)
    type_std = type_means.std(dim=0, keepdim=True) + 1e-8
    type_means = type_means / type_std

    # 相关系数矩阵 → (n_types, n_types)
    corr = (type_means.T @ type_means) / (batch_size - 1)

    # 惩罚非对角平方相关
    mask = 1 - torch.eye(n_types, device=gated_feats.device)
    ortho_loss = (corr * mask).pow(2).sum() / (n_types * (n_types - 1))

    return ortho_loss


# =============================================================================
# P2SignalAugmentedGatedMLP
# =============================================================================
class P2SignalAugmentedGatedMLP(nn.Module):
    """基于rel_enact+P2反模式信号的增强门控MLP。

    与标准GatedMLP的关键区别：
      1. 输入三路：relevance + enactment + P2_raw
      2. 全矩阵门控（396维×396维，含P2信号的调节）
      3. 分类层额外拼接类型级三路mean统计（5类型×3统计=15维）
         → 鸟瞰视角：某类型整体"高P2低enact"则应判无毒

    结构：
      gate = sigmoid(W @ x)  ← 全矩阵门控（x含P2信息）
      x_gated = x * gate
      summary = [mean_t(rel), mean_t(enact), mean_t(p2) for t in 5types]  ← 15维
      h = FC(concat[x_gated, summary] → 96) → ReLU → Dropout → FC(2)

    Args:
        n_concepts: 132 for v2
        concept_types: 概念类型列表
        dropout_rate: 标准0.5
        hidden_features: 标准96
    """

    def __init__(self, n_concepts, concept_types, dropout_rate=0.5, hidden_features=96,
                 n_summary=None, snr_bias_init=None, n_main_channels=3):
        super().__init__()
        self.n_concepts = n_concepts
        # 主特征总维度：n_channels × n_concepts（默认3路=P3+P2+contrast，含P1时为4路）
        self.main_dim = n_concepts * n_main_channels

        type_names = sorted(set(concept_types))
        self.n_types = len(type_names)  # 5 for v2
        # 记录每个类型的索引
        self.type_indices = {}
        for t in type_names:
            self.type_indices[t] = [i for i, ct in enumerate(concept_types) if ct == t]

        # 汇总特征维度：默认3统计/类型(mean only)，contrast_max模式为6统计/类型
        if n_summary is None:
            n_summary = self.n_types * 3  # 默认15

        # 全矩阵门控（含P2信号调节）
        self.gate_layer = nn.Linear(self.main_dim, self.main_dim)

        # SNR-informed gate bias initialization
        # 正SNR概念→正bias→门控默认开启（信号通过）
        # 负SNR概念→负bias→门控默认关闭（噪声抑制）
        if snr_bias_init is not None:
            with torch.no_grad():
                self.gate_layer.bias.copy_(torch.tensor(snr_bias_init, dtype=torch.float32))

        # 分类层：主特征(396) + 汇总(15) = 411 → hidden → 2
        total_dim = self.main_dim + n_summary
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(total_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x, return_gated=False):
        """
        Args:
            x: (batch, main_dim+summary) = 主特征 + 类型汇总
               注意：汇总特征放在末尾，门控只作用于主特征
            return_gated: 若True，额外返回门控后的主特征（用于COL等辅助损失）
        """
        main_feats = x[:, 0:self.main_dim]       # (B, main_dim)
        summary_feats = x[:, self.main_dim:]       # (B, n_summary)

        # 全矩阵门控（含P2的信号，sigmoid可学习反模式抑制）
        gate = torch.sigmoid(self.gate_layer(main_feats))
        x_gated = main_feats * gate

        # 拼接门控主特征 + 类型汇总统计
        fused = torch.cat([x_gated, summary_feats], dim=1)
        fused = self.dropout(fused)
        h = self.relu(self.fc1(fused))
        h = self.dropout(h)
        logits = self.fc2(h)
        if return_gated:
            return logits, x_gated
        return logits


# =============================================================================
# 特征提取
# =============================================================================
def extract_p2signal_features(data, concept_types, mode="p2signal", snr_weights=None, include_p1=False):
    """从3级概念向量提取特征矩阵。

    四种模式：

    mode="p2signal"（原始模式, 411维）:
      rel = P(2)+P(3)          ← 概念相关性（涉及+实施混合）
      enact = P(3)/(P(2)+P(3)) ← 实施概率（非线性比率，小值时不稳定）
      p2 = P(2)                ← 涉及信号
      类型汇总: (rel_mean, enact_mean, p2_mean) × 5类型

    mode="contrast"（对比模式, 411维）:
      p3 = P(3)                ← 实施信号（有毒方向，直接可用）
      p2 = P(2)                ← 涉及信号（无毒方向，直接可用）
      contrast = P(3) - P(2)   ← 涉及/实施对比（正=偏实施→有毒，负=偏涉及→无毒）
      类型汇总: (p3_mean, p2_mean, contrast_mean) × 5类型

    mode="contrast_max"（对比增强模式, 426维）:
      主特征同contrast，类型汇总增加max和正对比计数
      失败样本分析发现：个别概念P(3)很高但被均值稀释→加max；
      FP vs TN的关键区分是正对比概念数→加正对比计数
      类型汇总: (mean×3 + max×2 + count×1) × 5类型

    mode="contrast_snr"（SNR加权对比模式, 426维）:
      主特征: P(3) + P(2) + SNR_weighted_contrast
      contrast_snr = contrast × snr_weight
      SNR权重从训练数据计算：高SNR概念(如"嘲讽地"0.85)权重大，
      低SNR概念(如"狗哨政治的"-0.36)权重被压制为0
      → 少数误激活的低SNR概念噪声被压制，直接对症FP根因
      类型汇总: 同contrast_max（mean×3 + weighted_max×2 + weighted_count×1）

    Returns:
        X: (n_samples, n_features) 特征矩阵
        y: (n_samples,) 标签
    """
    n_samples = len(data)
    n_concepts = len(data[0]["concept"])
    type_names = sorted(set(concept_types))
    type_indices = {t: [i for i, ct in enumerate(concept_types) if ct == t] for t in type_names}

    # 提取原始3级概率
    p3_arr = np.zeros((n_samples, n_concepts))  # P(3) 实施
    p2_arr = np.zeros((n_samples, n_concepts))  # P(2) 涉及
    p1_arr = np.zeros((n_samples, n_concepts))  # P(1) 不相关
    y = np.zeros(n_samples, dtype=int)

    for si, item in enumerate(data):
        lp = item["level_probs"]
        for ci in range(n_concepts):
            p3_arr[si, ci] = lp[ci][2]  # P(3)
            p2_arr[si, ci] = lp[ci][1]  # P(2)
            p1_arr[si, ci] = lp[ci][0]  # P(1) = 1-P(2)-P(3)
        y[si] = item["toxic"]

    # 根据模式构建主特征和汇总
    if mode == "p2signal":
        # 原始模式: rel + enact + p2
        relevances = p2_arr + p3_arr
        enactments = p3_arr / (p2_arr + p3_arr + 1e-8)
        p2_raw = p2_arr
        main_features = [relevances, enactments, p2_raw]
        n_stats_per_type = 3  # mean(rel, enact, p2)
    elif mode == "contrast":
        # 对比模式: P(3) + P(2) + (P(3)-P(2))
        contrast = p3_arr - p2_arr
        main_features = [p3_arr, p2_arr, contrast]
        n_stats_per_type = 3  # mean(p3, p2, contrast)
    elif mode == "contrast_max":
        # 对比增强模式: 主特征同contrast，类型汇总增加max和正对比计数
        # 失败样本分析发现：
        #   1. FN样本中个别概念P(3)很高(如"嘲讽地"0.679)但被均值稀释 → 加max
        #   2. FP vs TN的关键区分是正对比概念数(8.2 vs 0.3) → 加正对比计数
        contrast = p3_arr - p2_arr
        main_features = [p3_arr, p2_arr, contrast]
        n_stats_per_type = 6  # mean(3) + max(2) + count(1)
    elif mode == "contrast_snr":
        # SNR加权对比模式: contrast × snr_weight
        # 深度分析发现：FP的根本原因是少数低SNR概念误激活P(3)>P(2)
        #   "令人反感的"(SNR=0.72)在25.7%的FP中误激活，主导了有毒判断
        #   即使全局contrast偏无毒(-0.4987)，门控仍被少数高权重概念翻转
        # 解决方案：用训练集SNR权重加权contrast，压制低SNR概念噪声
        contrast = p3_arr - p2_arr
        if snr_weights is None:
            raise ValueError("contrast_snr模式需要提供snr_weights参数")
        # snr_weights: (n_concepts,) 从训练数据计算的SNR权重
        # 权重设计: max(snr, 0) + eps → 负SNR概念权重为eps(极小正值，近乎零贡献)
        #                                    高SNR概念权重大(如"嘲讽地"0.85→权重0.85)
        snr_w = np.clip(snr_weights, 0, None) + 0.01  # eps=0.01确保负SNR概念近乎零但不为零
        weighted_contrast = contrast * snr_w  # (n_samples, n_concepts)
        main_features = [p3_arr, p2_arr, weighted_contrast]
        n_stats_per_type = 6  # mean(3) + max(2) + weighted_count(1)
    else:
        raise ValueError(f"未知特征模式: {mode}，支持: p2signal, contrast, contrast_max")

    # 可选P(1)通道：直接提供"概念不相关"信号
    # 即使P(1)=1-P(2)-P(3)线性相关，显式提供可节省门控学习成本
    if include_p1:
        main_features.append(p1_arr)
        n_stats_per_type += 2  # P1_mean(自动) + P1_max(手动)

    # 类型级汇总特征
    n_types = len(type_names)
    n_summary = n_types * n_stats_per_type
    summary = np.zeros((n_samples, n_summary))
    for si in range(n_samples):
        for t_idx, t in enumerate(type_names):
            idx = type_indices[t]
            base = t_idx * n_stats_per_type
            # mean统计（所有模式共有）
            for k, feats in enumerate(main_features):
                summary[si, base + k] = np.mean(feats[si, idx])
            # contrast_max和contrast_snr模式的额外统计
            if mode in ("contrast_max", "contrast_snr"):
                # main_features[2]是contrast（contrast_max）或weighted_contrast（contrast_snr）
                weighted_or_raw = main_features[2]
                extra_offset = 1 if include_p1 else 0  # P1_mean占用了一个位置
                summary[si, base + 3 + extra_offset] = np.max(p3_arr[si, idx])           # p3_max: 类型内最强实施信号
                summary[si, base + 4 + extra_offset] = np.max(weighted_or_raw[si, idx])  # contrast_max: 类型内最强对比(加权版)
                # 正对比计数：contrast_snr用加权后的阈值判断
                if mode == "contrast_max":
                    summary[si, base + 5 + extra_offset] = float(np.sum(contrast[si, idx] > 0))  # n_positive: P(3)>P(2)的概念数
                elif mode == "contrast_snr":
                    # 加权后的正对比：weighted_contrast > 0 且 snr_weight > 0.05（排除极低SNR概念）
                    summary[si, base + 5 + extra_offset] = float(np.sum(
                        (weighted_or_raw[si, idx] > 0) & (snr_w[idx] > 0.05)
                    ))  # n_high_snr_positive: 高SNR概念中P(3)>P(2)的数量
            # P(1)统计（include_p1时添加）
            if include_p1:
                summary[si, base + n_stats_per_type - 1] = np.max(p1_arr[si, idx])  # P1_max: 类型内最不相关概念

    X = np.concatenate(main_features + [summary], axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =============================================================================
# train_one_trial
# =============================================================================
def find_optimal_threshold(val_probs, val_labels, min_t=0.30, max_t=0.70, step=0.005):
    """在验证集上搜索最优决策阈值，最大化macro F1。

    默认argmax对应threshold=0.5。搜索范围0.30~0.70，步长0.005。
    降低阈值→更倾向判Toxic（TX recall↑，NT recall↓）
    升高阈值→更倾向判Non-Toxic（NT recall↑，TX recall↓）

    Returns: (best_threshold, best_val_f1)
    """
    best_t = 0.5
    best_f1 = 0.0
    thresholds = np.arange(min_t, max_t + step, step)
    for t in thresholds:
        preds = [1 if p[1] > t else 0 for p in val_probs]
        f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = round(t, 4)
    return best_t, best_f1


def train_one_trial(train_X, train_y, test_X, test_y, concept_types,
                    config=None, use_sam=False, sam_rho=0.05, trial_idx=0,
                    n_summary=None, optimize_threshold=False, snr_bias_init=None,
                    label_smoothing=0.0, n_main_channels=3, mixup_alpha=0.0,
                    col_lambda=0.0, n_concepts=132,
                    use_ema=False, ema_decay=0.999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = train_X.shape[1]  # 411

    # 全随机split（random_state=None），不固定任何seed
    # 保存split索引供事后复现
    all_indices = np.arange(len(train_X))
    tr_X_np, va_X_np, tr_y_np, va_y_np, tr_idx, va_idx = train_test_split(
        train_X.numpy(), train_y.numpy(), all_indices,
        test_size=0.2, stratify=train_y.numpy(), random_state=None,
    )
    tr_X = torch.tensor(tr_X_np)
    va_X = torch.tensor(va_X_np)
    tr_y = torch.tensor(tr_y_np)
    va_y = torch.tensor(va_y_np)

    train_loader = DataLoader(TensorDataset(tr_X, tr_y),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_X, va_y),
                            batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y),
                             batch_size=config.batch_size, shuffle=False)

    model = P2SignalAugmentedGatedMLP(
        n_concepts=n_concepts, concept_types=concept_types,
        dropout_rate=config.dropout_rate, hidden_features=config.hidden_features,
        n_summary=n_summary, snr_bias_init=snr_bias_init,
        n_main_channels=n_main_channels,
    ).to(device)

    # 完全标准配置
    # Mixup激活时自动禁用label_smoothing（mixup已提供足够正则化）
    effective_ls = label_smoothing if mixup_alpha == 0 else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=effective_ls)
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)
    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, total_steps=total_steps,
        pct_start=config.pct_start, anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor, final_div_factor=config.final_div_factor,
    )
    # EMA权重平均（可选）：维护参数的指数移动平均副本
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0
    history = {'epochs': [], 'val_f1': [], 'test_f1': []}

    pbar = tqdm(range(config.epochs), desc=f"Trial {trial_idx+1}{'(SAM)' if use_sam else ''}{'(mixup)' if mixup_alpha > 0 else ''}{'(COL)' if col_lambda > 0 else ''}{'(EMA)' if use_ema else ''}")
    for epoch in pbar:
        model.train()
        for batch in train_loader:
            bx, by = batch
            bx, by = bx.to(device), by.to(device)
            if use_sam:
                # SAM: 两次前向-反向，用扰动点梯度更新（不兼容mixup/COL）
                optimizer.zero_grad()
                sam_step(model, optimizer, criterion, bx, by, rho=sam_rho)
                optimizer.step()
                scheduler.step()
            elif mixup_alpha > 0:
                # Mixup: 特征空间插值 + 混合标签损失（不兼容COL）
                mixed_x, y_a, y_b, lam = mixup_data(bx, by, mixup_alpha)
                optimizer.zero_grad()
                pred = model(mixed_x)
                loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
                loss.backward()
                optimizer.step()
                scheduler.step()
            elif col_lambda > 0:
                # 标准训练 + COL: CE loss + λ × 类型正交损失
                optimizer.zero_grad()
                pred, gated = model(bx, return_gated=True)
                ce_loss = criterion(pred, by)
                ortho_loss = concept_orthogonal_loss(gated, concept_types, n_concepts)
                loss = ce_loss + col_lambda * ortho_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                # 标准训练: 一次前向-反向
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # EMA权重更新（在每个训练步后）
            if ema_model is not None:
                ema_model.update_parameters(model)

        # val（使用EMA权重如果已启用）
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                out = eval_model(bx.to(device))
                vp.extend(torch.argmax(out, dim=1).cpu().numpy())
                vl.extend(by.numpy())
        val_f1 = f1_score(vl, vp, average='macro')

        # test观察（使用EMA权重，不参与模型选择）
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                bx, by = batch
                out = eval_model(bx.to(device))
                tp.extend(torch.argmax(out, dim=1).cpu().numpy())
                tl.extend(by.numpy())
        test_f1 = f1_score(tl, tp, average='macro')

        history['epochs'].append(epoch + 1)
        history['val_f1'].append(val_f1)
        history['test_f1'].append(test_f1)
        pbar.set_postfix({'val_f1': f'{val_f1:.4f}', 'test_f1': f'{test_f1:.4f}', 'best': f'{best_f1:.4f}'})

        if val_f1 > best_f1:
            best_f1 = val_f1
            # 保存EMA权重（如果启用），否则保存原始权重
            state_source = ema_model.module if ema_model is not None else model
            best_state_dict = {k: v.clone() for k, v in state_source.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.patience:
            pbar.close()
            print(f">>> 早停: 连续{config.patience}个epoch未提升")
            break

    # 最终测试（仅一次）
    model.load_state_dict(best_state_dict)
    model.eval()

    # 获取测试集概率输出
    ap, al, apr = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            bx, by = batch
            out = model(bx.to(device))
            probs = torch.softmax(out, dim=1)
            ap.extend(torch.argmax(out, dim=1).cpu().numpy())
            al.extend(by.numpy())
            apr.extend(probs.cpu().numpy())

    final_f1 = f1_score(al, ap, average='macro')
    final_p = precision_score(al, ap, average='macro', zero_division=0)
    final_r = recall_score(al, ap, average='macro', zero_division=0)
    report = classification_report(al, ap, target_names=["Non-Toxic", "Toxic"])

    # 阈值优化（可选）：在验证集上搜索最优决策阈值
    opt_threshold = 0.5  # 默认
    opt_test_f1 = final_f1  # 默认等于argmax结果
    opt_test_report = ""
    opt_nt_recall = recall_score(al, ap, labels=[0], average=None)[0]
    opt_tx_recall = recall_score(al, ap, labels=[1], average=None)[0]

    if optimize_threshold:
        # 获取验证集概率输出
        vpr = []
        vlr = []
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                out = model(bx.to(device))
                probs = torch.softmax(out, dim=1)
                vpr.extend(probs.cpu().numpy())
                vlr.extend(by.numpy())

        # 搜索最优阈值
        opt_threshold, opt_val_f1 = find_optimal_threshold(vpr, vlr)
        # 应用最优阈值到测试集
        opt_preds = [1 if p[1] > opt_threshold else 0 for p in apr]
        opt_test_f1 = f1_score(al, opt_preds, average='macro')
        opt_test_p = precision_score(al, opt_preds, average='macro', zero_division=0)
        opt_test_r = recall_score(al, opt_preds, average='macro', zero_division=0)
        opt_nt_recall = recall_score(al, opt_preds, labels=[0], average=None)[0]
        opt_tx_recall = recall_score(al, opt_preds, labels=[1], average=None)[0]
        opt_test_report = classification_report(al, opt_preds, target_names=["Non-Toxic", "Toxic"])

        delta = opt_test_f1 - final_f1
        print(f"\n>>> 阈值优化: default(0.5) F1={final_f1:.4f} → optimal({opt_threshold:.3f}) F1={opt_test_f1:.4f}  Δ={delta:+.4f}")
        print(f"    NT_recall: {recall_score(al, ap, labels=[0], average=None)[0]:.4f} → {opt_nt_recall:.4f}")
        print(f"    TX_recall: {recall_score(al, ap, labels=[1], average=None)[0]:.4f} → {opt_tx_recall:.4f}")

    sam_tag = f", SAM(ρ={sam_rho})" if use_sam else ""
    print(f"\n>>> Trial {trial_idx+1}: best_epoch={best_epoch}, val_f1={best_f1:.4f}, test_f1={final_f1:.4f}{sam_tag}")
    print(f"    Precision={final_p:.4f}, Recall={final_r:.4f}")
    print(f"    Non-Toxic Recall = {recall_score(al, ap, labels=[0], average=None)[0]:.4f}")

    return {
        'trial_idx': trial_idx, 'best_epoch': best_epoch, 'best_val_f1': best_f1,
        'test_f1': final_f1, 'test_precision': final_p, 'test_recall': final_r,
        'nt_recall': recall_score(al, ap, labels=[0], average=None)[0],
        'tx_recall': recall_score(al, ap, labels=[1], average=None)[0],
        'use_sam': use_sam, 'sam_rho': sam_rho if use_sam else None,
        'split_info': {'train_idx': tr_idx.tolist(), 'val_idx': va_idx.tolist()},
        'opt_threshold': opt_threshold,
        'opt_test_f1': opt_test_f1,
        'opt_nt_recall': opt_nt_recall,
        'opt_tx_recall': opt_tx_recall,
        'opt_test_report': opt_test_report if optimize_threshold else "",
        'classification_report': report,
        'predictions': {'labels': al, 'preds': ap, 'probs': apr},
        'history': history, 'model_state_dict': best_state_dict,
    }


def plot_metrics(experiment_dir, history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['epochs'], history['val_f1'], label='Val F1', color='tab:blue')
    ax.plot(history['epochs'], history['test_f1'], label='Test F1 (obs)',
            color='tab:red', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Macro F1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('v2 3level P2Signal Training Metrics')
    plt.tight_layout()
    plt.savefig(experiment_dir / 'metrics.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="v2 3level 保守增量：rel_enact(264)+P2反模式(132)+轻量TypeAug汇总(15)=411维",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--use_sam', action='store_true',
                        help='开启SAM(Sharpness-Aware Minimization)优化器')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM扰动半径，默认0.05')
    parser.add_argument('--feature_mode', type=str, default='p2signal',
                        choices=['p2signal', 'contrast', 'contrast_max', 'contrast_snr'],
                        help='特征模式: p2signal(rel+enact+p2), contrast(P3+P2+P3-P2), '
                             'contrast_max(对比+类型max+正对比计数), '
                             'contrast_snr(SNR加权对比+类型max+高SNR正对比计数)')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='在验证集上搜索最优决策阈值并应用到测试集（不改模型架构，仅后处理）')
    parser.add_argument('--gate_snr_scale', type=float, default=0.0,
                        help='SNR-informed gate bias initialization scale. '
                             'Uses raw SNR as gate bias prior: positive-SNR→positive bias(gate opens), '
                             'negative-SNR→negative bias(gate suppresses). 0=no init(default). Recommended: 2.0-3.0')
    parser.add_argument('--include_p1', action='store_true',
                        help='Include P(1) (concept irrelevant probability) as 4th feature channel. '
                             'P(1)+P(2)+P(3)=1, provides direct "concept not present" signal. '
                             'Works with any --feature_mode.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing强度。0=硬标签(default)，推荐0.05-0.1。'
                             '将one-hot标签平滑为soft标签(如[0.95,0.05]替代[1,0])，'
                             '抑制模型过度自信和过拟合，适合小样本/高噪声场景。')
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Mixup特征空间插值强度。0=无mixup(default)，推荐0.2-0.4。'
                             'x̃=λxᵢ+(1-λ)xⱼ，ỹ=λyᵢ+(1-λ)yⱼ，等效数据增广。'
                             '开启后自动禁用label_smoothing（避免双重正则化）。')
    parser.add_argument('--col_lambda', type=float, default=0.0,
                        help='Concept Orthogonal Loss权重（NeurIPS 2023 coop-CBM）。'
                             '鼓励5种概念类型在门控后产生去相关的表示，减少信息冗余。'
                             '0=无COL(default)，推荐0.01-0.1。与mixup/SAM不兼容。')
    parser.add_argument('--snr_prune_threshold', type=float, default=None,
                        help='SNR概念剪枝阈值。SNR≤此阈值的概念从特征空间中移除。'
                             'None=不剪枝(default)，0.0=移除所有负SNR概念(约28个)。'
                             '剪枝减少维度+消除反向噪声，理论上提升信噪比。')
    parser.add_argument('--use_ema', action='store_true',
                        help='使用EMA(指数移动平均)权重进行验证和测试。'
                             '平滑训练噪声，提升泛化性和模型选择稳定性。')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA衰减率，默认0.999。越接近1平均窗口越长。')
    args = parser.parse_args()

    config = MLPConfig()
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name

    concept_train_path = (config.processed_path / args.dataset_name / args.model_name
                          / f"concept_train_{args.model_name}_v2_3level.json")
    concept_test_path = (config.processed_path / args.dataset_name / args.model_name
                         / f"concept_test_{args.model_name}_v2_3level.json")
    if not concept_train_path.exists() or not concept_test_path.exists():
        raise FileNotFoundError(f"概念向量不存在: {concept_train_path}")
    print(f">>> 加载v2 3level概念向量: {concept_train_path.parent}")

    with open(concept_train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(concept_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    n_concepts = len(train_data[0]["concept"])
    print(f">>> train={len(train_data)}, test={len(test_data)}, n_concepts={n_concepts}")

    type_mapping_path = config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json"
    with open(type_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    concept_types = [item["type"] for item in mapping]
    from collections import Counter
    type_dist = dict(Counter(concept_types))
    print(f">>> 类型分布: {type_dist}")

    # 计算SNR权重（contrast_snr模式、gate_snr_scale > 0、或剪枝时都需要）
    snr_weights = None
    snr_bias_init = None
    need_snr = (args.feature_mode == "contrast_snr") or (args.gate_snr_scale > 0) or (args.snr_prune_threshold is not None)
    if need_snr:
        # 从训练数据计算每个概念的contrast SNR
        train_toxic = [item for item in train_data if item['toxic'] == 1]
        train_non_toxic = [item for item in train_data if item['toxic'] == 0]
        snr_weights = np.zeros(n_concepts)
        for ci in range(n_concepts):
            toxic_contrast = [item['level_probs'][ci][2] - item['level_probs'][ci][1]
                              for item in train_toxic]
            nt_contrast = [item['level_probs'][ci][2] - item['level_probs'][ci][1]
                           for item in train_non_toxic]
            mean_diff = np.mean(toxic_contrast) - np.mean(nt_contrast)
            pooled_std = np.std(toxic_contrast + nt_contrast) + 1e-8
            snr_weights[ci] = mean_diff / pooled_std
        # 打印SNR统计
        pos_snr = np.sum(snr_weights > 0)
        neg_snr = np.sum(snr_weights < 0)
        print(f">>> SNR权重: 正SNR概念={pos_snr}, 负SNR概念={neg_snr}")
        print(f">>> SNR分布: mean={np.mean(snr_weights):.4f} max={np.max(snr_weights):.4f} min={np.min(snr_weights):.4f}")
        # 打印Top 5高SNR概念
        top_snr_idx = np.argsort(snr_weights)[-5:][::-1]
        for ci in top_snr_idx:
            print(f"    #{ci} {concept_types[ci]}/{mapping[ci]['chinese']}: SNR={snr_weights[ci]:.4f}")

        # SNR驱动的概念剪枝：移除SNR≤阈值的概念
        if args.snr_prune_threshold is not None:
            keep_mask = snr_weights > args.snr_prune_threshold
            n_pruned = int((~keep_mask).sum())
            # 打印移除的概念
            removed_idx = np.where(~keep_mask)[0]
            removed_snr = snr_weights[~keep_mask]
            sort_idx = np.argsort(removed_snr)
            print(f"\n>>> 概念剪枝(阈值>{args.snr_prune_threshold}): 保留{keep_mask.sum()}/{n_concepts}个, 移除{n_pruned}个")
            for ri in removed_idx[sort_idx][:min(8, n_pruned)]:
                print(f"    ✂ #{ri} {concept_types[ri]}/{mapping[ri]['chinese']}: SNR={snr_weights[ri]:.4f}")

            # 剪枝概念类型和数据
            concept_types = [ct for i, ct in enumerate(concept_types) if keep_mask[i]]
            type_dist = dict(Counter(concept_types))
            print(f">>> 剪枝后类型分布: {type_dist}")

            for item in train_data:
                item["level_probs"] = [lp for i, lp in enumerate(item["level_probs"]) if keep_mask[i]]
                item["concept"] = [c for i, c in enumerate(item["concept"]) if keep_mask[i]]
            for item in test_data:
                item["level_probs"] = [lp for i, lp in enumerate(item["level_probs"]) if keep_mask[i]]
                item["concept"] = [c for i, c in enumerate(item["concept"]) if keep_mask[i]]

            n_concepts = int(keep_mask.sum())
            snr_weights = snr_weights[keep_mask]  # 同步剪枝SNR权重

        # SNR-informed gate bias initialization
        if args.gate_snr_scale > 0:
            # 使用原始SNR值（非截断）作为gate bias先验
            # 正SNR → 正bias → sigmoid(bias)>0.5 → 门控开启（概念信号通过）
            # 负SNR → 负bias → sigmoid(bias)<0.5 → 门控关闭（噪声概念抑制）
            # N维 = n_main_channels × n_concepts，每概念n_main_channels个维度共享同一SNR值
            snr_bias_init = np.tile(snr_weights, n_main_channels) * args.gate_snr_scale
            print(f">>> Gate SNR初始化: scale={args.gate_snr_scale}, "
                  f"bias范围=[{np.min(snr_bias_init):.4f}, {np.max(snr_bias_init):.4f}]")
            n_open = int(np.sum(snr_bias_init > 0))
            n_suppress = int(np.sum(snr_bias_init < 0))
            print(f"    Gate维度: 开启(正bias)={n_open}, 抑制(负bias)={n_suppress}")

    train_X, train_y = extract_p2signal_features(train_data, concept_types,
                                                  mode=args.feature_mode, snr_weights=snr_weights,
                                                  include_p1=args.include_p1)
    test_X, test_y = extract_p2signal_features(test_data, concept_types,
                                                mode=args.feature_mode, snr_weights=snr_weights,
                                                include_p1=args.include_p1)
    in_features = train_X.shape[1]
    n_main_channels = 4 if args.include_p1 else 3
    feat_desc_map = {
        "p2signal": "rel+enact+p2",
        "contrast": "P3+P2+(P3-P2)",
        "contrast_max": "P3+P2+(P3-P2)+类型max+正对比计数",
        "contrast_snr": "P3+P2+SNR加权(P3-P2)+类型max+高SNR正对比计数",
    }
    feat_desc = feat_desc_map[args.feature_mode]
    if args.include_p1:
        feat_desc += "+P1"
    # 计算类型汇总维度
    n_stats_per_type = 6 if args.feature_mode in ("contrast_max", "contrast_snr") else 3
    if args.include_p1:
        n_stats_per_type += 2  # P1_mean(自动) + P1_max(手动)
    n_summary = len(type_dist) * n_stats_per_type
    print(f">>> 特征模式: {args.feature_mode}{' + P1' if args.include_p1 else ''} ({feat_desc})")
    print(f">>> 特征维度: {in_features} = 主特征({n_concepts}×{n_main_channels}={n_concepts*n_main_channels}) + 类型级汇总({n_summary})")
    if args.snr_prune_threshold is not None:
        print(f">>> SNR剪枝: 阈值>{args.snr_prune_threshold}, 概念{132}→{n_concepts}")
    print(f">>> 模型: P2SignalAugmentedGatedMLP (全矩阵门控 + 类型汇总注入分类层)")
    if args.gate_snr_scale > 0:
        print(f">>> Gate SNR初始化: scale={args.gate_snr_scale} (正SNR→正bias开启, 负SNR→负bias抑制)")
    print(f">>> 训练: 标准配置 AdamW+OneCycleLR batch={config.batch_size} dropout={config.dropout_rate} patience={config.patience}")
    if args.label_smoothing > 0:
        print(f">>> Label Smoothing: {args.label_smoothing} (soft label, 抑制过度自信)")
    if args.mixup_alpha > 0:
        print(f">>> Mixup: alpha={args.mixup_alpha} (特征插值数据增广)")
        if args.label_smoothing > 0:
            print(f"    (mixup激活 → label_smoothing自动禁用，避免双重正则化)")
    if args.col_lambda > 0:
        print(f">>> COL(lambda={args.col_lambda}): 类型正交损失 (coop-CBM, NeurIPS 2023)")
    if args.use_ema:
        print(f">>> EMA: decay={args.ema_decay} (指数移动平均权重，用于val/test评估)")
    if args.use_sam:
        print(f">>> SAM已开启: rho={args.sam_rho} (两次前向-反向/step，训练时间约2x)")

    all_results = []
    for trial_idx in range(args.n_trials):
        print("\n" + "=" * 55)
        print(f"  Trial {trial_idx + 1}/{args.n_trials} (全随机split)")
        print("=" * 55)
        r = train_one_trial(train_X, train_y, test_X, test_y,
                            concept_types=concept_types, config=config,
                            use_sam=args.use_sam, sam_rho=args.sam_rho,
                            trial_idx=trial_idx, n_summary=n_summary,
                            optimize_threshold=args.optimize_threshold,
                            snr_bias_init=snr_bias_init,
                            label_smoothing=args.label_smoothing,
                            n_main_channels=n_main_channels,
                            mixup_alpha=args.mixup_alpha,
                            col_lambda=args.col_lambda,
                            n_concepts=n_concepts,
                            use_ema=args.use_ema, ema_decay=args.ema_decay)
        all_results.append(r)

    all_results.sort(key=lambda r: r["best_val_f1"], reverse=True)
    best_trial = all_results[0]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = config.experiment_path / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "timestamp": timestamp,
        "pipeline": "v2_3level_p2signal",
        "model_type": "P2SignalAugmentedGatedMLP",
        "feature_mode": args.feature_mode,
        "n_features": in_features,
        "n_concepts": n_concepts,
        "type_distribution": type_dist,
        "dataset_name": args.dataset_name, "model_name": args.model_name,
        "seed": "random (探索阶段不固定)", "batch_size": config.batch_size,
        "epochs": config.epochs, "max_lr": config.max_lr,
        "dropout_rate": config.dropout_rate, "hidden_features": config.hidden_features,
        "patience": config.patience,
        "label_smoothing": args.label_smoothing, "include_p1": args.include_p1, "mixup_alpha": args.mixup_alpha,
        "col_lambda": args.col_lambda,
        "snr_prune_threshold": args.snr_prune_threshold,
        "use_ema": args.use_ema, "ema_decay": args.ema_decay if args.use_ema else None,
        "use_sam": args.use_sam, "sam_rho": args.sam_rho if args.use_sam else None,
        "gate_snr_scale": args.gate_snr_scale,
        "n_trials": args.n_trials, "best_trial_idx": best_trial["trial_idx"],
    }
    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    torch.save(best_trial["model_state_dict"], experiment_dir / "best_model.pth")
    plot_metrics(experiment_dir, best_trial['history'])

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "pipeline": "v2_3level_p2signal",
        "model_type": "P2SignalAugmentedGatedMLP",
        "n_features": in_features,
        "use_sam": args.use_sam, "sam_rho": args.sam_rho if args.use_sam else None,
        "best_trial_idx": best_trial['trial_idx'],
        "best_epoch": best_trial['best_epoch'],
        "best_val_f1": round(best_trial['best_val_f1'], 4),
        "precision_macro": round(best_trial['test_precision'], 4),
        "recall_macro": round(best_trial['test_recall'], 4),
        "f1_macro": round(best_trial['test_f1'], 4),
    }
    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("v2 3level P2反模式 分类器评估结果\n")
        f.write("=" * 35 + "\n")
        f.write(f"模型: P2SignalAugmentedGatedMLP (rel+enact+P2 + 类型汇总)\n")
        f.write(f"特征维度: {in_features}\n")
        if args.use_sam:
            f.write(f"SAM: rho={args.sam_rho}\n")
        f.write(f"最佳Trial: #{best_trial['trial_idx']+1}\n")
        f.write(f"最佳Epoch: {best_trial['best_epoch']}\n")
        f.write(f"Val F1: {best_trial['best_val_f1']:.4f}\n")
        f.write(f"Test F1: {best_trial['test_f1']:.4f}\n")
        f.write("-" * 35 + "\n")
        f.write(best_trial['classification_report'])

    preds_obj = best_trial['predictions']
    predictions = []
    for i in range(len(preds_obj['labels'])):
        predictions.append({
            "index": i,
            "true_label": int(preds_obj['labels'][i]),
            "pred_label": int(preds_obj['preds'][i]),
            "correct": bool(preds_obj['labels'][i] == preds_obj['preds'][i]),
            "probabilities": {
                "non_toxic": round(float(preds_obj['probs'][i][0]), 4),
                "toxic": round(float(preds_obj['probs'][i][1]), 4),
            },
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # 保存全部trial明细（trials_summary.json），供事后分析对比
    trials_summary = {
        "config": {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "n_trials": args.n_trials,
            "architecture": f"P2SignalAugmentedGatedMLP {in_features}d [{args.feature_mode}{'+P1' if args.include_p1 else ''}]" + (f" + SAM(rho={args.sam_rho})" if args.use_sam else ""),
            "loss": f"CrossEntropyLoss(label_smoothing={args.label_smoothing})" if (args.label_smoothing > 0 and args.mixup_alpha == 0) else ("MixupCELoss" if args.mixup_alpha > 0 else "CrossEntropyLoss"),
            "optimizer_scheduler": "AdamW + OneCycleLR",
            "batch_size": config.batch_size,
            "epochs_max": config.epochs,
            "dropout": config.dropout_rate,
            "hidden_features": config.hidden_features,
            "patience": config.patience,
            "label_smoothing": args.label_smoothing,
            "include_p1": args.include_p1,
            "mixup_alpha": args.mixup_alpha,
            "col_lambda": args.col_lambda,
        "snr_prune_threshold": args.snr_prune_threshold,
        "use_ema": args.use_ema, "ema_decay": args.ema_decay if args.use_ema else None,
            "use_sam": args.use_sam,
            "sam_rho": args.sam_rho if args.use_sam else None,
            "gate_snr_scale": args.gate_snr_scale,
            "seed_policy": "fully_random (random_state=None, no fixed seed)",
            "optimize_threshold": args.optimize_threshold,
            "type_distribution": type_dist,
        },
        "trials": [],
        "summary": {},
    }
    for r in all_results:
        trial_data = {
            "idx": r["trial_idx"] + 1,
            "best_epoch": r["best_epoch"],
            "val_f1_best": round(r["best_val_f1"], 4),
            "test_f1": round(r["test_f1"], 4),
            "test_precision": round(r["test_precision"], 4),
            "test_recall": round(r["test_recall"], 4),
            "nt_recall": round(r["nt_recall"], 4),
            "tx_recall": round(r["tx_recall"], 4),
        }
        if args.optimize_threshold:
            trial_data["opt_threshold"] = r["opt_threshold"]
            trial_data["opt_test_f1"] = round(r["opt_test_f1"], 4)
            trial_data["opt_nt_recall"] = round(r["opt_nt_recall"], 4)
            trial_data["opt_tx_recall"] = round(r["opt_tx_recall"], 4)
        trials_summary["trials"].append(trial_data)

    test_f1s = [r['test_f1'] for r in all_results]
    val_f1s = [r['best_val_f1'] for r in all_results]
    summary_data = {
        "test_f1_mean": round(float(np.mean(test_f1s)), 4),
        "test_f1_std": round(float(np.std(test_f1s)), 4),
        "test_f1_max": round(float(np.max(test_f1s)), 4),
        "test_f1_min": round(float(np.min(test_f1s)), 4),
        "val_f1_mean": round(float(np.mean(val_f1s)), 4),
        "val_f1_std": round(float(np.std(val_f1s)), 4),
    }
    if args.optimize_threshold:
        opt_f1s = [r['opt_test_f1'] for r in all_results]
        summary_data["opt_test_f1_mean"] = round(float(np.mean(opt_f1s)), 4)
        summary_data["opt_test_f1_std"] = round(float(np.std(opt_f1s)), 4)
        summary_data["opt_test_f1_max"] = round(float(np.max(opt_f1s)), 4)
        summary_data["opt_test_f1_min"] = round(float(np.min(opt_f1s)), 4)
    trials_summary["summary"] = summary_data
    with open(experiment_dir / "trials_summary.json", "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, ensure_ascii=False)

    # 保存全部trial的split索引（供论文阶段固定seed复现）
    splits_info = {}
    for r in all_results:
        splits_info[f"trial_{r['trial_idx']+1}"] = r['split_info']
    with open(experiment_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    sam_label = f" + SAM(ρ={args.sam_rho})" if args.use_sam else ""
    ls_label = f" + LS{args.label_smoothing}" if args.label_smoothing > 0 else ""
    mx_label = f" + mixup{args.mixup_alpha}" if args.mixup_alpha > 0 else ""
    col_label = f" + COL{args.col_lambda}" if args.col_lambda > 0 else ""
    prune_label = f" (pruned>{args.snr_prune_threshold})" if args.snr_prune_threshold is not None else ""
    ema_label = f" + EMA{args.ema_decay}" if args.use_ema else ""
    p1_label = " + P1" if args.include_p1 else ""
    feat_label = f" [{args.feature_mode}{p1_label}]"
    print(f"  v2 3level P2Signal 汇总 ({args.n_trials} trials{feat_label}{ls_label}{mx_label}{col_label}{prune_label}{ema_label}{sam_label})")
    print("=" * 60)
    print(f"{'Trial':<8}{'Best Ep':<10}{'Val F1':<10}{'Test F1':<10}{'NT Rcl':<10}{'TX Rcl':<10}")
    if args.optimize_threshold:
        print(f"{'':8}{'':10}{'':10}{'Opt F1':<10}{'Opt NT':<10}{'Opt TX':<10}{'Opt T':<8}")
    print("-" * 58)
    for r in all_results:
        line = f"#{r['trial_idx']+1:<7}{r['best_epoch']:<10}{r['best_val_f1']:<10.4f}{r['test_f1']:<10.4f}{r['nt_recall']:<10.4f}{r['tx_recall']:<10.4f}"
        if args.optimize_threshold:
            line += f"{r['opt_test_f1']:<10.4f}{r['opt_nt_recall']:<10.4f}{r['opt_tx_recall']:<10.4f}{r['opt_threshold']:<8.3f}"
        print(line)
    print("-" * 58)
    print(f"  Test F1 mean = {np.mean(test_f1s):.4f}  std = {np.std(test_f1s):.4f}  best = {np.max(test_f1s):.4f}  worst = {np.min(test_f1s):.4f}")
    if args.optimize_threshold:
        opt_f1s = [r['opt_test_f1'] for r in all_results]
        print(f"  Opt  F1 mean = {np.mean(opt_f1s):.4f}  std = {np.std(opt_f1s):.4f}  best = {np.max(opt_f1s):.4f}  worst = {np.min(opt_f1s):.4f}")
        print(f"  阈值优化均值提升: Δ = {np.mean(opt_f1s) - np.mean(test_f1s):+.4f}")
    print(f"  选中Trial #{best_trial['trial_idx']+1}: F1={best_trial['test_f1']:.4f}")
    print(best_trial['classification_report'])
    if args.optimize_threshold and best_trial.get('opt_test_f1') != best_trial['test_f1']:
        print(f"\n  阈值优化版(t={best_trial['opt_threshold']:.3f}): F1={best_trial['opt_test_f1']:.4f}")
        # 打印阈值优化版的分类报告（如果有）
        if 'opt_test_report' in best_trial and best_trial['opt_test_report']:
            print(best_trial['opt_test_report'])
    print(f"\n>>> 结果: {experiment_dir}")
    print(f">>> trials_summary.json: {experiment_dir / 'trials_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
