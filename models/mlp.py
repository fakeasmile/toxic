import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96,
                 gate_type='matrix', gate_init=None):
        """
        概念瓶颈分类器，支持三种门控模式。

        Args:
            in_features: 输入特征维度（概念向量维度）
            dropout_rate: dropout 比率
            hidden_features: 隐藏层维度
            gate_type: 门控类型
                - 'matrix': 原始矩阵门控 sigmoid(W@x+b)，样本相关（参数量 in_features^2）
                - 'global': 全局门控 sigmoid(v)，所有样本共享重要性权重（参数量 in_features）
                - 'none': 无门控
            gate_init: 门控初始化向量（numpy 数组，长度=in_features），传入 Cohen's d 作为统计先验。
                - gate_type='global': 直接作为门控向量 v 初始化
                - gate_type='matrix': 作为门控偏置 b 初始化，W 保持随机初始化学习样本相关调整
        """
        super(MLP, self).__init__()
        self.gate_type = gate_type

        # ========== 门控单元 ==========
        if gate_type == 'matrix':
            # 原始矩阵门控：gate = sigmoid(W @ x + b)，样本相关
            self.gate_layer = nn.Linear(in_features, in_features)  # 形状：[A, A]
            if gate_init is not None:
                # 用 Cohen's d 初始化偏置 b：d>0→门控默认打开(有毒特征)，d<0→门控默认关闭(正向概念)
                # W 保持随机初始化，学习样本相关的调整
                init_val = torch.tensor(gate_init, dtype=torch.float32)
                with torch.no_grad():
                    self.gate_layer.bias.copy_(init_val)
        elif gate_type == 'global':
            # 全局门控：gate = sigmoid(v)，所有样本共享同一重要性权重
            # gate_init: 传入 Cohen's d 向量用于初始化，d>0→门控打开(有毒特征)，d<0→门控关闭(正向概念)
            if gate_init is not None:
                init_val = torch.tensor(gate_init, dtype=torch.float32)
                self.gate_vector = nn.Parameter(init_val.clone())
            else:
                self.gate_vector = nn.Parameter(torch.zeros(in_features))
        elif gate_type == 'none':
            pass  # 无门控

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, feature_vector, return_gate=False):
        inputs = feature_vector

        # 门控
        if self.gate_type == 'matrix':
            gate_weights = torch.sigmoid(self.gate_layer(feature_vector))
            x = inputs * gate_weights
        elif self.gate_type == 'global':
            gate_weights = torch.sigmoid(self.gate_vector)
            # 扩展到 batch 维度，与 matrix 门控输出形状一致 [batch_size, in_features]
            gate_weights = gate_weights.unsqueeze(0).expand_as(inputs)
            x = inputs * gate_weights
        else:  # 'none'
            gate_weights = None
            x = inputs

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_gate:
            return x, gate_weights
        return x

    def get_gate_l1_loss(self):
        """返回门控参数的 L1 正则化损失，训练时加入总 loss 强制稀疏性。"""
        if self.gate_type == 'global':
            return self.gate_vector.abs().sum()
        elif self.gate_type == 'matrix':
            return self.gate_layer.weight.abs().sum()
        return torch.tensor(0.0, device=self.fc1.weight.device)
