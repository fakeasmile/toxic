import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        # [新增] 门控机制：学习每个特征的重要性权重 (0-1)
        self.gate_layer = nn.Linear(in_features, in_features)

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, feature_vector):
        x = feature_vector

        # [新增] 应用门控逻辑
        # 使用 Sigmoid 激活函数将权重限制在 0 到 1 之间
        gate_weights = torch.sigmoid(self.gate_layer(x))
        # 逐元素相乘，强化重要特征，抑制无关特征
        x = x * gate_weights

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x