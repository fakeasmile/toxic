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


class PCE_MLP(nn.Module):
    """概率概念嵌入MLP（Probabilistic Concept Embedding MLP）

    将Likert 5级概率分布通过可学习嵌入矩阵转化为概念嵌入向量，
    替代原始SCBM中的固定权重加权期望标量。

    核心改动：
        当前SCBM:  p(1..5) → score = Σ w_k · p(k)     → 1个标量/概念
        PCE:       p(1..5) → c = Σ E_k · p(k)           → m维向量/概念

    其中 E ∈ R^{5×m} 是可学习的等级嵌入矩阵，p(k) 是LLM给出的Likert概率。
    """

    def __init__(self, num_concepts, embed_dim=16, dropout_rate=0.5, hidden_features=128):
        """
        Args:
            num_concepts: 形容词概念数量（如177）
            embed_dim: 每个概念的嵌入维度
            dropout_rate: Dropout比率
            hidden_features: 分类隐藏层维度
        """
        super(PCE_MLP, self).__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim

        # Likert 5级等级的可学习嵌入矩阵（全局共享）
        # E[k] 表示第k级Likert等级的嵌入表示
        self.level_embeddings = nn.Parameter(torch.randn(5, embed_dim) * 0.01)

        in_features = num_concepts * embed_dim

        # 门控单元
        self.gate_layer = nn.Linear(in_features, in_features)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)

        # 分类层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_features, 2)

    def forward(self, likert_probs):
        """
        Args:
            likert_probs: (batch, V, 5) 每个形容词的5级Likert概率分布
        Returns:
            logits: (batch, 2) 二分类输出
        """
        # 概念嵌入: c_i = Σ_k p(k) · E_k
        # (batch, V, 5) @ (5, m) = (batch, V, m)
        concept_embeds = torch.matmul(likert_probs, self.level_embeddings)

        # 展平: (batch, V*m)
        x = concept_embeds.view(concept_embeds.size(0), -1)

        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(x))
        x = x * gate_weights

        # Dropout + 分类
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x