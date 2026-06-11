"""CB-LLM-CN模型定义

Concept Bottleneck LLM for Chinese Toxic Language Detection

核心组件：
1. 非线性概念瓶颈层（CBL）：将backbone嵌入映射到概念空间，使用两层MLP+ReLU防止线性退化
2. TopK稀疏激活：只保留TopK个最相关概念，强制概念稀疏性
3. 稀疏线性预测层：L1正则化确保每个预测只依赖少量关键概念
4. 可选残差连接：从backbone到预测层的跳跃连接，防止信息瓶颈

参考：CB-LLM (ICLR 2025), CBM-Suite (CVPR 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBLLM_CN(nn.Module):
    """CB-LLM-CN: Concept Bottleneck LLM for Chinese Toxic Language Detection

    Args:
        num_concepts: 概念数量 K（如177）
        backbone_dim: backbone嵌入维度（如768 for BGE）
        cbl_hidden_dim: CBL中间层维度（如256）
        cbl_dropout: CBL Dropout比率（如0.3）
        sparse_l1_weight: L1稀疏正则化权重（如0.01）
        prediction_dropout: 预测层Dropout（如0.3）
        topk: TopK稀疏激活数量，0表示不使用TopK（如30）
        use_residual: 是否使用残差连接（backbone→预测层跳跃连接）
    """

    def __init__(self, num_concepts, backbone_dim=768, cbl_hidden_dim=256,
                 cbl_dropout=0.3, sparse_l1_weight=0.01, prediction_dropout=0.3,
                 topk=0, use_residual=False):
        super(CBLLM_CN, self).__init__()

        self.num_concepts = num_concepts
        self.backbone_dim = backbone_dim
        self.sparse_l1_weight = sparse_l1_weight
        self.topk = topk
        self.use_residual = use_residual

        # ========== 非线性概念瓶颈层（CBL）==========
        # 两层MLP + ReLU，防止CBM-Suite指出的"线性退化"问题
        self.cbl = nn.Sequential(
            nn.Linear(backbone_dim, cbl_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cbl_dropout),
            nn.Linear(cbl_hidden_dim, num_concepts),
            nn.ReLU(),  # ReLU确保概念激活非负
        )

        # ========== 稀疏线性预测层 ==========
        # 线性层确保可解释性：每个预测可追溯到具体概念
        self.prediction_layer = nn.Linear(num_concepts, 2)
        self.prediction_dropout = nn.Dropout(prediction_dropout)

        # ========== 可选残差连接 ==========
        # 从backbone到预测层的跳跃连接，防止信息瓶颈
        # 使用低秩投影保持概念层的主导地位
        self.residual_layer = None
        if use_residual:
            self.residual_layer = nn.Sequential(
                nn.Linear(backbone_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

    def forward(self, backbone_embeds, concept_targets=None):
        """前向传播

        Args:
            backbone_embeds: [B, backbone_dim] backbone模型嵌入
            concept_targets: [B, K] 概念目标（训练时用于CBL监督）

        Returns:
            logits: [B, 2] 分类logits
            concept_activations: [B, K] 概念激活值
            l1_loss: L1稀疏正则化损失
        """
        # ========== 概念瓶颈层 ==========
        concept_activations = self.cbl(backbone_embeds)  # [B, K]

        # ========== TopK稀疏激活 ==========
        if self.topk > 0 and self.topk < self.num_concepts:
            # 保留TopK个最大激活，其余置零
            topk_values, topk_indices = concept_activations.topk(self.topk, dim=1)
            mask = torch.zeros_like(concept_activations)
            mask.scatter_(1, topk_indices, 1.0)
            concept_activations = concept_activations * mask

        # ========== 稀疏预测 ==========
        x = self.prediction_dropout(concept_activations)
        logits = self.prediction_layer(x)  # [B, 2]

        # ========== 残差连接 ==========
        if self.residual_layer is not None:
            residual_logits = self.residual_layer(backbone_embeds)
            logits = logits + residual_logits

        # ========== L1稀疏正则化 ==========
        l1_loss = self.sparse_l1_weight * torch.abs(self.prediction_layer.weight).sum()

        return logits, concept_activations, l1_loss

    def compute_loss(self, logits, labels, concept_activations, concept_targets, l1_loss):
        """计算总损失

        L_total = L_cls + L_cbl + L_sparse

        Args:
            logits: [B, 2] 分类logits
            labels: [B] 标签
            concept_activations: [B, K] CBL输出的概念激活
            concept_targets: [B, K] 概念目标
            l1_loss: L1稀疏正则化损失

        Returns:
            total_loss: 总损失
            loss_dict: 各损失项的字典
        """
        # 分类损失
        L_cls = F.cross_entropy(logits, labels)

        # CBL概念对齐损失：MSE between CBL output and concept targets
        L_cbl = F.mse_loss(concept_activations, concept_targets)

        # 总损失
        total_loss = L_cls + L_cbl + l1_loss

        loss_dict = {
            "L_cls": L_cls.item(),
            "L_cbl": L_cbl.item(),
            "L_sparse": l1_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def get_concept_contributions(self, concept_activations):
        """计算每个概念对分类的贡献评分

        Args:
            concept_activations: [B, K] 概念激活值

        Returns:
            contributions: [B, K, 2] 每个概念对每个类别的贡献
            top_concepts: [B, K] 每个概念的总贡献绝对值
        """
        contributions = concept_activations.unsqueeze(-1) * self.prediction_layer.weight.t().unsqueeze(0)
        top_concepts = contributions.abs().sum(dim=-1)
        return contributions, top_concepts
