"""PCCG: Pragmatic Causal Concept Graph — 语用因果概念图模型

架构：
  BGE嵌入(7×768) + Likert标量(81维)
      ↓
  因果概念图 (GAT消息传递)
      ↓
  分组信息瓶颈 (每节点独立IB约束)
      ↓
  概念得分投影 (768→1 per node)
      ↓
  拼接Likert标量 → 稀疏门控分类器 → 有毒/无毒
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """单头图注意力层

    对小图（7个节点）使用简化版GAT，无需稀疏矩阵操作。
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, edge_index):
        """
        Args:
            h: [B, N, in_dim] 节点特征
            edge_index: [2, E] 边索引 (src, dst)

        Returns:
            [B, N, out_dim] 更新后的节点特征
        """
        B, N, _ = h.shape
        Wh = self.W(h)  # [B, N, out_dim]

        # 计算注意力系数
        # e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
        e_src = self.a_src(Wh).squeeze(-1)  # [B, N]
        e_dst = self.a_dst(Wh).squeeze(-1)  # [B, N]

        # 对每条边计算注意力
        src_idx, dst_idx = edge_index[0], edge_index[1]
        # e_{src->dst}
        e = self.leaky_relu(e_src[:, src_idx] + e_dst[:, dst_idx])  # [B, E]

        # 对每个目标节点做softmax（按dst分组）
        # 构建注意力矩阵 [B, N, N]，初始化为-inf
        attn = torch.full((B, N, N), float('-inf'), device=h.device)
        attn[:, dst_idx, src_idx] = e
        attn = F.softmax(attn, dim=-1)  # [B, N, N]，每行对src做softmax

        # 将-inf位置softmax后的0/NaN替换为0
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # 消息传递：h' = Σ_j α_ij * Wh_j
        out = torch.bmm(attn, Wh)  # [B, N, out_dim]
        return out


class MultiHeadGATLayer(nn.Module):
    """多头图注意力层"""

    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_dim, out_dim // num_heads) for _ in range(num_heads)
        ])
        assert out_dim % num_heads == 0, f"out_dim({out_dim})必须能被num_heads({num_heads})整除"

    def forward(self, h, edge_index):
        head_outputs = [head(h, edge_index) for head in self.heads]
        return torch.cat(head_outputs, dim=-1)


class CausalConceptGraph(nn.Module):
    """因果概念图：GAT消息传递

    7个节点（7个语用推理维度），通过因果边进行消息传递。
    """

    def __init__(self, node_dim, hidden_dim, num_heads=4, num_layers=2):
        super().__init__()
        # 节点编码器：各节点输入维度相同(768) → hidden_dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads))

        # 节点解码器：hidden_dim → 原始维度
        self.node_decoder = nn.Linear(hidden_dim, node_dim)

    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [B, N, node_dim] 各节点的BGE嵌入
            edge_index: [2, E] 因果边

        Returns:
            [B, N, node_dim] GNN消息传递后的节点表示
        """
        h = self.node_encoder(node_features)  # [B, N, hidden_dim]

        for gat in self.gat_layers:
            h_residual = h
            h = gat(h, edge_index)
            h = F.elu(h)
            h = h + h_residual  # 残差连接

        output = self.node_decoder(h)  # [B, N, node_dim]
        return output


class GroupedIB(nn.Module):
    """分组信息瓶颈

    对因果图中每个节点独立施加变分IB约束，防止信息泄漏。
    每个节点有独立的μ和logvar线性层。
    """

    def __init__(self, node_dim, num_nodes, ib_beta=1.0):
        super().__init__()
        self.ib_beta = ib_beta
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        # 每个节点独立的变分编码器
        self.mu_layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim) for _ in range(num_nodes)
        ])
        self.logvar_layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim) for _ in range(num_nodes)
        ])

    def forward(self, node_features, training=True):
        """
        Args:
            node_features: [B, N, node_dim] GNN输出

        Returns:
            output: [B, N, node_dim] IB约束后的节点表示
            ib_loss: 标量，总KL散度损失
        """
        B, N, D = node_features.shape
        total_kl = 0.0
        outputs = []

        for i in range(N):
            feat = node_features[:, i, :]  # [B, D]
            mu = self.mu_layers[i](feat)           # [B, D]
            logvar = self.logvar_layers[i](feat)    # [B, D]

            if training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu

            # KL散度: -0.5 * sum(1 + log(σ²) - μ² - σ²)，对batch维度取均值
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            total_kl += kl
            outputs.append(z)

        output = torch.stack(outputs, dim=1)  # [B, N, D]
        return output, self.ib_beta * total_kl


class PCCG(nn.Module):
    """PCCG: 语用因果概念图模型

    完整流程：
    1. BGE嵌入 → 因果图GAT消息传递
    2. 分组IB约束
    3. 每节点投影为标量概念得分
    4. 拼接Likert标量 → 稀疏门控分类器
    """

    def __init__(
        self,
        num_nodes=7,
        bge_dim=768,
        gnn_hidden_dim=256,
        gnn_num_heads=4,
        gnn_num_layers=2,
        ib_beta=1.0,
        num_likert_concepts=81,
        hidden_features=128,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.bge_dim = bge_dim
        self.num_likert_concepts = num_likert_concepts

        # 1. 因果概念图
        self.causal_graph = CausalConceptGraph(
            node_dim=bge_dim,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            num_layers=gnn_num_layers,
        )

        # 2. 分组信息瓶颈
        self.grouped_ib = GroupedIB(
            node_dim=bge_dim,
            num_nodes=num_nodes,
            ib_beta=ib_beta,
        )

        # 3. 概念得分投影：每个节点768维 → 1维标量
        self.concept_score_projections = nn.ModuleList([
            nn.Linear(bge_dim, 1) for _ in range(num_nodes)
        ])

        # 4. 稀疏门控分类器
        # 输入：7个因果概念得分 + 81个Likert标量 = 88维
        total_concepts = num_nodes + num_likert_concepts
        self.gate_layer = nn.Linear(total_concepts, total_concepts)
        self.fc1 = nn.Linear(total_concepts, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, bge_embeddings, likert_scores, edge_index):
        """
        Args:
            bge_embeddings: [B, N, bge_dim] 7个维度的BGE嵌入
            likert_scores: [B, num_likert_concepts] Likert标量概念向量
            edge_index: [2, E] 因果边

        Returns:
            logits: [B, 2] 分类logits
            gate_weights: [B, total_concepts] 门控权重
            ib_loss: 标量，IB损失
            concept_scores: [B, N] 因果概念得分
        """
        # 1. 因果图消息传递
        graph_output = self.causal_graph(bge_embeddings, edge_index)  # [B, N, bge_dim]

        # 2. 分组IB约束
        ib_output, ib_loss = self.grouped_ib(graph_output)  # [B, N, bge_dim]

        # 3. 因果概念得分投影
        concept_scores = torch.cat([
            self.concept_score_projections[i](ib_output[:, i, :])
            for i in range(self.num_nodes)
        ], dim=-1)  # [B, N]

        # 4. 拼接Likert标量
        total_concepts = torch.cat([concept_scores, likert_scores], dim=-1)  # [B, N+81]

        # 5. 稀疏门控分类
        gate_weights = torch.sigmoid(self.gate_layer(total_concepts))
        x = total_concepts * gate_weights
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, gate_weights, ib_loss, concept_scores
