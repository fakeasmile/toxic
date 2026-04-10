import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, dropout_rate=0.2, hidden_features=96):
        super(MLP, self).__init__()
        # ========== 门控单元，学习每个特征的重要性权重 (0-1) ==========
        self.gate_layer = nn.Linear(in_features, in_features)  # 形状：[A, A]

        # ========== 归一化 ==========
        self.norm = nn.LayerNorm(in_features)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, feature_vector):
        inputs = feature_vector

        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(feature_vector))
        x = inputs * gate_weights

        # 归一化+dropout
        # x = self.norm(x)
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x