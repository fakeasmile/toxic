import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96):
        """
        概念瓶颈分类器，使用矩阵门控。

        结构: 门控(sigmoid(W@x)) → Dropout → FC(in→hidden) → ReLU → Dropout → FC(hidden→2)

        Args:
            in_features: 输入特征维度（概念向量维度）
            dropout_rate: dropout 比率
            hidden_features: 隐藏层维度
        """
        super(MLP, self).__init__()

        # ========== 门控单元 ==========
        # 矩阵门控：gate = sigmoid(W @ x)，样本相关
        self.gate_layer = nn.Linear(in_features, in_features)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, feature_vector):
        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(feature_vector))
        x = feature_vector * gate_weights

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TypeAugmentedGatedMLP(nn.Module):
    """类型增强门控MLP：保留全矩阵门控 + 类型级汇总特征。

    在标准GatedMLP基础上，为分类层添加每个概念类型的统计汇总特征
    (mean, max)，使模型在细粒度概念分数之外，还能利用粗粒度的
    类型级信息。这为分类器提供了"鸟瞰"视角。

    结构:
      gate = sigmoid(W @ x)       ← 全矩阵门控（同GatedMLP）
      x_gated = x * gate
      summary = [mean_t(x_gated), max_t(x_gated) for t in types]  ← 7类型×2统计=14维
      h = FC(concat[x_gated, summary] → hidden) → ReLU → Dropout → FC(2)

    Args:
        in_features: 输入特征维度
        concept_types: 概念类型列表
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, concept_types, dropout_rate=0.5, hidden_features=96):
        super(TypeAugmentedGatedMLP, self).__init__()

        # 记录每种类型的索引
        self.type_names = sorted(set(concept_types))
        self.type_indices = {}
        for t in self.type_names:
            self.type_indices[t] = [i for i, ct in enumerate(concept_types) if ct == t]

        # 类型级汇总特征维度: 7类型 × 2统计(mean, max)
        n_summary = len(self.type_names) * 2

        # 全矩阵门控（同GatedMLP）
        self.gate_layer = nn.Linear(in_features, in_features)

        # 分类层（输入维度 = 概念维度 + 汇总维度）
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features + n_summary, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 全矩阵门控（同GatedMLP）
        gate_weights = torch.sigmoid(self.gate_layer(x))
        x_gated = x * gate_weights

        # 类型级汇总特征
        summary_parts = []
        for t in self.type_names:
            idx = self.type_indices[t]
            type_feats = x_gated[:, idx]
            summary_parts.append(type_feats.mean(dim=1, keepdim=True))
            summary_parts.append(type_feats.max(dim=1, keepdim=True)[0])
        summary = torch.cat(summary_parts, dim=1)

        # 拼接门控特征 + 汇总特征
        x = torch.cat([x_gated, summary], dim=1)

        # 分类
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x