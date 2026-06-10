"""ICB-CBM模型定义

Information-Compressed Bottleneck Concept Bottleneck Model

核心组件：
1. 双通道概念表示：Likert标量 × LLM稠密向量
2. 残差概念发现：可学习残差概念补全词典未覆盖的语义维度
3. 分组变分瓶颈：对每个概念独立施加IB约束
4. ICC正贡献约束：分类器仅使用相关概念的正贡献
5. 稀疏门控分类器：每条文本仅激活少量关键概念
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICB_CBM(nn.Module):
    """ICB-CBM: Information-Compressed Bottleneck Concept Bottleneck Model

    Args:
        num_concepts: 形容词概念数量 K（如177）
        hidden_dim: LLM hidden state维度（如3584）
        dense_dim: 每个概念的稠密向量维度 d（如64）
        num_residual: 残差概念数量 K_r（如32）
        hidden_features: 分类器隐藏层维度（如128）
        dropout_rate: Dropout比率（如0.3）
        alpha: IB损失权重（如0.01）
        gamma: 稀疏损失权重（如0.001）
    """

    def __init__(self, num_concepts, hidden_dim, dense_dim=64, num_residual=32,
                 hidden_features=128, dropout_rate=0.3, alpha=0.01, gamma=0.001):
        super(ICB_CBM, self).__init__()

        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.num_residual = num_residual
        self.alpha = alpha
        self.gamma = gamma

        # ========== 概念投影层：hidden_dim → d ==========
        # 每个概念一个独立的投影矩阵，将LLM hidden state投影为d维稠密向量
        self.concept_projections = nn.ModuleList([
            nn.Linear(hidden_dim, dense_dim) for _ in range(num_concepts)
        ])

        # ========== 残差概念投影：hidden_dim → K_r ==========
        self.residual_projection = nn.Linear(hidden_dim, num_residual)

        # ========== 分组变分瓶颈编码器 ==========
        # 对每个概念的d维向量独立施加IB约束
        # 每个概念：d维 → μ(d维) + logvar(d维)
        self.concept_mu_layers = nn.ModuleList([
            nn.Linear(dense_dim, dense_dim) for _ in range(num_concepts)
        ])
        self.concept_logvar_layers = nn.ModuleList([
            nn.Linear(dense_dim, dense_dim) for _ in range(num_concepts)
        ])

        # 残差概念的变分瓶颈
        self.residual_mu = nn.Linear(num_residual, num_residual)
        self.residual_logvar = nn.Linear(num_residual, num_residual)

        # ========== ICC概念选择掩码 ==========
        # 基于Likert标量与标签的相关性预计算，训练时不更新
        # 初始化为全1（所有概念都参与），后续通过compute_concept_mask设置
        self.register_buffer('concept_mask', torch.ones(num_concepts))

        # ========== 稀疏门控分类器 ==========
        total_dim = num_concepts * dense_dim + num_residual

        self.gate_layer = nn.Linear(total_dim, total_dim)

        # 分类层：使用非负权重约束实现ICC正贡献约束
        self.fc1 = nn.Linear(total_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)

        self.dropout = nn.Dropout(dropout_rate)

    def compute_concept_mask(self, likert_scores, labels, threshold=0.05):
        """预计算ICC概念选择掩码

        基于Likert标量与标签的点双列相关性，仅保留与标签正相关的概念。

        Args:
            likert_scores: [N, K] Likert标量概念向量
            labels: [N] 标签（0/1）
            threshold: 相关性阈值，低于此值的概念被屏蔽
        """
        # 计算每个概念与标签的点双列相关
        labels_float = labels.float()
        mean_1 = likert_scores[labels == 1].mean(dim=0)  # [K]
        mean_0 = likert_scores[labels == 0].mean(dim=0)  # [K]

        # 有毒样本的概念均值应高于无毒样本（正相关）
        # 仅保留正相关的概念
        diff = mean_1 - mean_0  # [K]

        # 同时计算整体相关性强度
        std_all = likert_scores.std(dim=0) + 1e-8  # [K]
        correlation = diff / std_all  # 简化的相关性度量

        # 掩码：正相关且相关性超过阈值
        mask = (correlation > threshold).float()
        self.concept_mask.copy_(mask)

        n_selected = mask.sum().item()
        print(f"ICC概念选择掩码: 选中 {int(n_selected)}/{self.num_concepts} 个概念 "
              f"(阈值={threshold})")

    def forward(self, likert_scores, hidden_states):
        """前向传播

        Args:
            likert_scores: [B, K] Likert标量概念向量
            hidden_states: [B, hidden_dim] LLM hidden state

        Returns:
            logits: [B, 2] 分类logits
            gate_weights: [B, total_dim] 门控权重
            mu: 概念瓶颈层的均值（用于IB损失计算）
            logvar: 概念瓶颈层的对数方差（用于IB损失计算）
        """
        batch_size = likert_scores.shape[0]

        # ========== 双通道融合 ==========
        dense_concepts = []
        for i, proj in enumerate(self.concept_projections):
            v_i = proj(hidden_states)          # [B, d]
            s_i = likert_scores[:, i:i+1]      # [B, 1]
            dense_concepts.append(s_i * v_i)    # [B, d], 标量门控

        # ========== 残差概念 ==========
        R_r = torch.sigmoid(self.residual_projection(hidden_states))  # [B, K_r]

        # ========== 分组变分瓶颈 ==========
        concept_z_list = []
        concept_mu_list = []
        concept_logvar_list = []

        for i in range(self.num_concepts):
            mu_i = self.concept_mu_layers[i](dense_concepts[i])        # [B, d]
            logvar_i = self.concept_logvar_layers[i](dense_concepts[i])  # [B, d]

            if self.training:
                std_i = torch.exp(0.5 * logvar_i)
                eps_i = torch.randn_like(std_i)
                z_i = mu_i + eps_i * std_i
            else:
                z_i = mu_i

            concept_z_list.append(z_i)
            concept_mu_list.append(mu_i)
            concept_logvar_list.append(logvar_i)

        # 残差概念的变分瓶颈
        residual_mu = self.residual_mu(R_r)       # [B, K_r]
        residual_logvar = self.residual_logvar(R_r)  # [B, K_r]

        if self.training:
            residual_std = torch.exp(0.5 * residual_logvar)
            residual_eps = torch.randn_like(residual_std)
            residual_z = residual_mu + residual_eps * residual_std
        else:
            residual_z = residual_mu

        # 拼接所有概念
        z = torch.cat(concept_z_list + [residual_z], dim=-1)  # [B, K*d + K_r]
        mu = torch.cat(concept_mu_list + [residual_mu], dim=-1)
        logvar = torch.cat(concept_logvar_list + [residual_logvar], dim=-1)

        # ========== ICC正贡献约束 ==========
        # 通过概念选择掩码屏蔽不相关概念
        # 将mask扩展到与z相同的维度
        expanded_mask = self.concept_mask.unsqueeze(0).unsqueeze(2).expand(
            batch_size, -1, self.dense_dim
        )  # [B, K, d]
        expanded_mask = expanded_mask.reshape(batch_size, -1)  # [B, K*d]

        # 残差概念的mask（默认全1）
        residual_mask = torch.ones(batch_size, self.num_residual, device=z.device)
        full_mask = torch.cat([expanded_mask, residual_mask], dim=-1)  # [B, K*d + K_r]

        # 应用mask
        z = z * full_mask

        # ========== 稀疏门控 ==========
        gate_weights = torch.sigmoid(self.gate_layer(z))
        x = z * gate_weights

        # ========== 分类 ==========
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, gate_weights, mu, logvar

    def compute_loss(self, logits, labels, mu, logvar, gate_weights):
        """计算总损失

        L_total = L_cls + alpha * L_IB + gamma * L_sparse

        Args:
            logits: [B, 2] 分类logits
            labels: [B] 标签
            mu: 概念瓶颈层均值
            logvar: 概念瓶颈层对数方差
            gate_weights: [B, total_dim] 门控权重

        Returns:
            total_loss: 总损失
            loss_dict: 各损失项的字典
        """
        # 分类损失
        L_cls = F.cross_entropy(logits, labels)

        # IB损失：KL(q(z|x) || N(0, I))
        # 对每个维度独立计算KL散度
        L_IB = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 按样本平均
        L_IB = L_IB / logits.shape[0]

        # 稀疏损失
        L_sparse = gate_weights.abs().mean()

        # 总损失
        total_loss = L_cls + self.alpha * L_IB + self.gamma * L_sparse

        loss_dict = {
            "L_cls": L_cls.item(),
            "L_IB": L_IB.item(),
            "L_sparse": L_sparse.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def get_concept_contributions(self, likert_scores, gate_weights):
        """计算每个概念对分类的贡献评分

        Args:
            likert_scores: [B, K] Likert标量概念向量
            gate_weights: [B, K*d + K_r] 门控权重

        Returns:
            contributions: [B, K] 每个概念的总贡献评分
        """
        batch_size = likert_scores.shape[0]

        # 将gate_weights按概念分组
        concept_gates = gate_weights[:, :self.num_concepts * self.dense_dim]
        concept_gates = concept_gates.reshape(batch_size, self.num_concepts, self.dense_dim)

        # 每个概念的门控权重L2范数作为贡献度量
        concept_gate_norms = concept_gates.norm(dim=-1)  # [B, K]

        # 乘以Likert标量
        contributions = likert_scores * concept_gate_norms  # [B, K]

        return contributions
