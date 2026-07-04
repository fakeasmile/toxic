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
