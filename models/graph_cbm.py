"""Graph-Structured Concept Bottleneck Model (GraphCBM)

核心创新：将扁平的177维概念向量转换为图结构，用GCN沿语义关系传播信息。

与之前方案的区别：
- 无残差连接从RoBERTa引入信息 → 无信息泄漏路径
- GCN仅在概念空间内传播信息 → 概念层不被架空
- 图结构来自语言学知识（形容词语义类别）→ 强归纳偏置

架构：
  原始概念向量(177维) → 节点特征扩展(1→hidden) → GCN图卷积 → 概念精炼
  → 门控 + MLP → 有毒/无毒分类
"""

import json
import torch
import torch.nn as nn
from pathlib import Path


def build_adjacency_matrix(category_path, num_adjectives=177, intra_weight=1.0, inter_weight=0.3):
    """从形容词类别文件构建归一化邻接矩阵。

    图结构：
    - 同类别内的形容词之间有边（权重=intra_weight）
    - 相邻类别的形容词之间有边（权重=inter_weight）
    - 自环（对角线=1）

    归一化：D^{-1/2} A D^{-1/2}（对称归一化，标准GCN做法）

    Args:
        category_path: adjective_categories.json 路径
        num_adjectives: 形容词数量（默认177）
        intra_weight: 同类别内边的权重
        inter_weight: 相邻类别间边的权重

    Returns:
        adj_norm: [num_adjectives, num_adjectives] 归一化邻接矩阵
    """
    with open(category_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    adj_categories = data["adjective_categories"]
    category_adjacency = data["category_adjacency"]

    # 构建形容词→类别映射
    adj_to_cats = {}
    for cat_name, adj_indices in adj_categories.items():
        for idx in adj_indices:
            if idx not in adj_to_cats:
                adj_to_cats[idx] = []
            adj_to_cats[idx].append(cat_name)

    # 构建原始邻接矩阵
    A = torch.zeros(num_adjectives, num_adjectives)

    # 同类别内边
    for cat_name, adj_indices in adj_categories.items():
        for i in adj_indices:
            for j in adj_indices:
                if i != j:
                    A[i][j] = max(A[i][j], intra_weight)

    # 相邻类别间边
    for cat_name, neighbor_cats in category_adjacency.items():
        if cat_name not in adj_categories:
            continue
        for neighbor_cat in neighbor_cats:
            if neighbor_cat not in adj_categories:
                continue
            for i in adj_categories[cat_name]:
                for j in adj_categories[neighbor_cat]:
                    A[i][j] = max(A[i][j], inter_weight)

    # 添加自环
    A = A + torch.eye(num_adjectives)

    # 对称归一化: D^{-1/2} A D^{-1/2}
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    adj_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return adj_norm


class GCNLayer(nn.Module):
    """单层图卷积: H = σ(A * X * W)"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj_norm):
        """
        Args:
            x: [batch, N, in_features] 节点特征
            adj_norm: [N, N] 归一化邻接矩阵
        Returns:
            [batch, N, out_features]
        """
        support = torch.matmul(x, self.weight)  # [batch, N, out_features]
        output = torch.matmul(adj_norm, support) + self.bias  # [batch, N, out_features]
        return torch.relu(output)


class GraphCBM(nn.Module):
    """Graph-Structured Concept Bottleneck Model

    流程：
    1. 原始概念向量(177维) → 节点特征(1维) → 扩展为hidden维
    2. GCN图卷积：沿形容词语义关系传播信息
    3. 概念精炼：GCN输出投影回1维，作为对原始分数的修正
    4. 门控 + MLP分类

    关键设计：
    - GCN修正量加在原始分数上（残差连接在概念空间内，不引入外部信息）
    - 门控层学习每个精炼后概念的重要性
    - 分类器仅从概念空间预测，架构级保证防信息泄漏
    """

    def __init__(self, num_concepts, adj_norm, gcn_hidden=32, mlp_hidden=96, dropout_rate=0.5, num_gcn_layers=2):
        super().__init__()
        self.num_concepts = num_concepts

        # 注册邻接矩阵为buffer（不参与梯度更新）
        self.register_buffer('adj_norm', adj_norm)

        # 节点特征扩展: 1 -> gcn_hidden
        self.node_encoder = nn.Linear(1, gcn_hidden)

        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(gcn_hidden, gcn_hidden))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(gcn_hidden, gcn_hidden))

        # 概念精炼投影: gcn_hidden -> 1
        self.concept_readout = nn.Linear(gcn_hidden, 1)

        # 门控单元
        self.gate_layer = nn.Linear(num_concepts, num_concepts)

        # 分类MLP
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_concepts, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 2)
        self.relu = nn.ReLU()

    def forward(self, concept_vector):
        """
        Args:
            concept_vector: [batch, 177] 原始形容词概念向量
        Returns:
            logits: [batch, 2] 分类logits
        """
        # 节点特征扩展
        x = concept_vector.unsqueeze(-1)  # [batch, 177, 1]
        h = self.node_encoder(x)  # [batch, 177, gcn_hidden]

        # GCN图卷积
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, self.adj_norm)  # [batch, 177, gcn_hidden]

        # 概念精炼：投影回1维，作为修正量
        gcn_correction = self.concept_readout(h).squeeze(-1)  # [batch, 177]
        refined = concept_vector + gcn_correction  # 残差修正（概念空间内）

        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(refined))
        x = refined * gate_weights

        # 分类
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_concept_importance(self, concept_vector):
        """获取概念重要性（用于可解释性分析）

        Returns:
            refined_scores: [batch, 177] 精炼后的概念分数
            gate_weights: [batch, 177] 门控权重
            combined: [batch, 177] 综合重要性 = refined * gate
        """
        with torch.no_grad():
            x = concept_vector.unsqueeze(-1)
            h = self.node_encoder(x)
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, self.adj_norm)
            gcn_correction = self.concept_readout(h).squeeze(-1)
            refined = concept_vector + gcn_correction
            gate_weights = torch.sigmoid(self.gate_layer(refined))
            combined = refined * gate_weights

        return refined, gate_weights, combined
