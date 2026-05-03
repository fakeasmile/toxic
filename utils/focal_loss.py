"""Focal Loss实现。

用于解决类别不平衡问题，让模型更关注难分样本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    公式: FL(pt) = -(1-pt)^gamma * log(pt)
    其中 pt = p 如果 y=1, pt = 1-p 如果 y=0

    gamma=0时退化为标准CrossEntropy
    gamma越大，对易分样本的抑制越强
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        :param gamma: 聚焦参数，越大对易分样本抑制越强
        :param alpha: 类别权重，用于处理类别不平衡
        :param reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型输出logits [B, num_classes]
        :param targets: 真实标签 [B]
        :return: focal loss标量
        """
        # 计算交叉熵损失（不reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 计算pt（模型对正确类别的预测概率）
        pt = torch.exp(-ce_loss)

        # 计算focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 根据reduction方式聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
