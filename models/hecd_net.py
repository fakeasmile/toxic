import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SimpleGATLayer(nn.Module):
    """
    简化版图注意力层 (Graph Attention Layer)
    不依赖 torch_geometric，纯 PyTorch 实现
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.0, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.head_dim = out_dim // num_heads

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: [N, in_dim] 节点特征
        adj: [N, N] 邻接矩阵 (0/1 或 float 权重)
        """
        N = x.size(0)
        # 线性变换 + 分头
        h = self.linear(x)                     # [N, out_dim]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        h = self.dropout(h)

        # 计算注意力分数
        # [N, num_heads, head_dim] * [1, num_heads, head_dim]
        attn_src = (h * self.att_src).sum(dim=-1).unsqueeze(1)   # [N, 1, num_heads]
        attn_dst = (h * self.att_dst).sum(dim=-1).unsqueeze(0)   # [1, N, num_heads]

        # [N, 1, num_heads] + [1, N, num_heads] -> [N, N, num_heads]
        attn_scores = attn_src + attn_dst
        attn_scores = self.leaky_relu(attn_scores)

        # Mask: 只在邻接矩阵有边的地方保留分数
        mask = (adj == 0).unsqueeze(-1)  # [N, N, 1]
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax 归一化 (对邻居维度)
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, N, num_heads]
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        # 消息传递: [N, N, num_heads] @ [N, num_heads, head_dim] -> [N, num_heads, head_dim]
        out = torch.einsum('nsh,shd->nhd', attn_weights, h)

        if self.concat:
            out = out.reshape(N, self.out_dim)  # [N, out_dim]
        else:
            out = out.mean(dim=1)               # [N, out_dim]

        return out + self.bias


class ConceptEmbeddingBank(nn.Module):
    """
    概念嵌入库: 每个概念是一个可学习的 concept_dim 维向量
    """
    def __init__(self, num_concepts, concept_dim):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self):
        return self.embeddings  # [num_concepts, concept_dim]


class CrossAttentionConceptActivation(nn.Module):
    """
    交叉注意力: h_cls 作为 Query，概念嵌入作为 Key/Value
    """
    def __init__(self, hidden_dim, concept_dim, num_concepts, dropout=0.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.key_proj = nn.Linear(concept_dim, hidden_dim)
        self.value_proj = nn.Linear(concept_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_cls, concept_embeddings):
        """
        h_cls: [B, hidden_dim]
        concept_embeddings: [num_concepts, concept_dim]
        """
        # 投影
        concept_keys = self.key_proj(concept_embeddings)      # [N, hidden_dim]
        concept_values = self.value_proj(concept_embeddings)  # [N, hidden_dim]

        # 交叉注意力分数
        attn_scores = torch.matmul(h_cls, concept_keys.t()) / self.scale  # [B, N]
        attn_weights = F.softmax(attn_scores, dim=-1)                     # [B, N]
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        concept_activated = torch.matmul(attn_weights, concept_values)    # [B, hidden_dim]

        return concept_activated, attn_weights


class HierarchicalConceptGraph(nn.Module):
    """
    层次化概念图: 使用 GAT 在概念节点间传播信息
    """
    def __init__(self, concept_dim, hidden_dim, num_gat_layers=2, num_heads=4, dropout=0.0):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_gat_layers = num_gat_layers

        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_d = concept_dim if i == 0 else hidden_dim
            out_d = hidden_dim
            self.gat_layers.append(
                SimpleGATLayer(in_d, out_d, num_heads=num_heads, dropout=dropout, concat=True)
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])
        self.output_proj = nn.Linear(hidden_dim, concept_dim)

    def forward(self, concept_embeddings, adj):
        """
        concept_embeddings: [N, concept_dim]
        adj: [N, N]
        """
        x = concept_embeddings
        for i, gat in enumerate(self.gat_layers):
            x = gat(x, adj)
            x = self.norms[i](x)
            x = F.elu(x)
        x = self.output_proj(x)
        return x


class PlatformTopicAdapter(nn.Module):
    """
    平台-主题适配器: 将平台和主题信息注入文本表示
    """
    def __init__(self, num_platforms, num_topics, adapter_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.platform_emb = nn.Embedding(num_platforms, adapter_dim)
        self.topic_emb = nn.Embedding(num_topics, adapter_dim)
        self.adapter_fusion = nn.Linear(adapter_dim * 2, adapter_dim)
        self.adapter_proj = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_cls, platform_ids, topic_ids):
        """
        h_cls: [B, hidden_dim]
        platform_ids: [B]
        topic_ids: [B]
        """
        platform_bias = self.platform_emb(platform_ids)   # [B, adapter_dim]
        topic_bias = self.topic_emb(topic_ids)            # [B, adapter_dim]
        fused = torch.cat([platform_bias, topic_bias], dim=-1)  # [B, 2*adapter_dim]
        fused = self.dropout(F.relu(self.adapter_fusion(fused)))  # [B, adapter_dim]
        adapter_output = self.adapter_proj(fused)         # [B, hidden_dim]
        return h_cls + adapter_output


class HECDNet(nn.Module):
    """
    HECD-Net: Hierarchical Embedding-based Concept Decomposition Network
    """
    def __init__(
        self,
        plm_name="chinese-roberta-wwm-ext",
        num_concepts=56,
        num_classes=2,
        num_topics=4,
        num_expressions=4,
        num_targets=5,
        num_platforms=2,
        plm_hidden_size=768,
        concept_dim=64,
        num_gat_layers=2,
        gat_hidden_dim=128,
        gat_num_heads=4,
        adapter_dim=64,
        dropout=0.3,
        use_residual=True,
        use_graph=True,
        use_adapter=True,
        use_auxiliary=True,
        concept_graph_path=None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.use_graph = use_graph
        self.use_adapter = use_adapter
        self.use_auxiliary = use_auxiliary
        self.plm_hidden_size = plm_hidden_size
        self.concept_dim = concept_dim

        # PLM Encoder
        self.plm = AutoModel.from_pretrained(plm_name)

        # 概念嵌入库
        self.concept_bank = ConceptEmbeddingBank(num_concepts, concept_dim)

        # 交叉注意力
        self.cross_attn = CrossAttentionConceptActivation(
            plm_hidden_size, concept_dim, num_concepts, dropout
        )

        # 层次化概念图
        if use_graph:
            self.concept_graph = HierarchicalConceptGraph(
                concept_dim, gat_hidden_dim, num_gat_layers, gat_num_heads, dropout
            )
            # 加载并缓存邻接矩阵
            self.register_buffer("adj_matrix", self._build_adjacency(concept_graph_path, num_concepts))
        else:
            self.concept_graph = None
            self.adj_matrix = None

        # 平台-主题适配器
        if use_adapter:
            self.adapter = PlatformTopicAdapter(
                num_platforms, num_topics, adapter_dim, plm_hidden_size, dropout
            )
        else:
            self.adapter = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 分类器输入维度
        classifier_input_dim = plm_hidden_size
        if use_residual:
            classifier_input_dim += plm_hidden_size  # h_cls + concept_activated

        # 主分类头: toxic (2-class)
        self.head_toxic = nn.Linear(classifier_input_dim, num_classes)

        # 辅助头
        if use_auxiliary:
            self.head_topic = nn.Linear(classifier_input_dim, num_topics)
            self.head_expression = nn.Linear(classifier_input_dim, num_expressions)
            self.head_target = nn.Linear(classifier_input_dim, num_targets)
        else:
            self.head_topic = None
            self.head_expression = None
            self.head_target = None

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _build_adjacency(self, concept_graph_path, num_concepts):
        """从 JSON 构建邻接矩阵"""
        if concept_graph_path is None or not Path(concept_graph_path).exists():
            # 默认全连接（包含自环），确保 GAT 有意义
            return torch.ones(num_concepts, num_concepts)

        with open(concept_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        adj = torch.zeros(num_concepts, num_concepts)
        for edge in graph_data.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            if 0 <= src < num_concepts and 0 <= tgt < num_concepts:
                adj[src, tgt] = 1.0

        # 加自环
        adj = adj + torch.eye(num_concepts)
        adj = (adj > 0).float()
        return adj

    def forward(
        self,
        input_ids,
        attention_mask,
        platform_ids=None,
        topic_ids=None,
        labels=None,
        topic_labels=None,
        expression_labels=None,
        target_labels=None,
    ):
        """
        前向传播
        """
        # PLM 编码
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        h_cls = self.dropout(h_cls)

        # 平台-主题适配
        if self.use_adapter and platform_ids is not None and topic_ids is not None:
            h_cls = self.adapter(h_cls, platform_ids, topic_ids)

        # 概念嵌入
        concept_emb = self.concept_bank()  # [N, concept_dim]

        # 概念图传播
        if self.use_graph and self.concept_graph is not None:
            device = h_cls.device
            adj = self.adj_matrix.to(device)
            concept_emb = self.concept_graph(concept_emb, adj)  # [N, concept_dim]

        # 交叉注意力激活
        concept_activated, attn_weights = self.cross_attn(h_cls, concept_emb)
        # attn_weights: [B, N]  <- 可解释性来源

        # 分类器输入
        if self.use_residual:
            classifier_input = torch.cat([h_cls, concept_activated], dim=-1)
        else:
            classifier_input = concept_activated

        # 主任务: toxic
        logits_toxic = self.head_toxic(classifier_input)

        # 辅助任务
        aux_logits = {}
        if self.use_auxiliary:
            aux_logits["topic"] = self.head_topic(classifier_input)
            aux_logits["expression"] = self.head_expression(classifier_input)
            aux_logits["target"] = self.head_target(classifier_input)

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.ce_loss(logits_toxic, labels)

            if self.use_auxiliary:
                if topic_labels is not None and self.head_topic is not None and self.head_topic.out_features > 1:
                    loss = loss + 0.3 * self.ce_loss(aux_logits["topic"], topic_labels)
                if expression_labels is not None and self.head_expression is not None and self.head_expression.out_features > 1:
                    loss = loss + 0.3 * self.ce_loss(aux_logits["expression"], expression_labels)
                if target_labels is not None and self.head_target is not None and self.head_target.out_features > 1:
                    loss = loss + 0.3 * self.bce_loss(aux_logits["target"], target_labels.float())

        outputs_dict = {
            "logits_toxic": logits_toxic,
            "attn_weights": attn_weights,
            "concept_activated": concept_activated,
            "h_cls": h_cls,
        }
        if aux_logits:
            outputs_dict["aux_logits"] = aux_logits

        if loss is not None:
            outputs_dict["loss"] = loss
            return outputs_dict

        return outputs_dict

    def get_explanation(self, input_ids, attention_mask, platform_ids=None, topic_ids=None):
        """
        获取可解释输出: 概念激活权重 + 推理路径
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask,
                platform_ids=platform_ids, topic_ids=topic_ids
            )

        attn_weights = outputs["attn_weights"]  # [B, N]
        # 返回每个样本激活最高的 Top-K 概念索引和权重
        topk_vals, topk_indices = torch.topk(attn_weights, k=5, dim=-1)

        return {
            "top_concepts": topk_indices.cpu().numpy(),
            "top_weights": topk_vals.cpu().numpy(),
            "all_weights": attn_weights.cpu().numpy(),
        }


def contrastive_concept_loss(attn_weights, labels, temperature=0.1):
    """
    对比概念学习损失
    attn_weights: [B, N]
    labels: [B]
    """
    B = attn_weights.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=attn_weights.device)

    # 归一化概念权重用于对比
    features = F.normalize(attn_weights, dim=-1)  # [B, N]

    # 相似度矩阵
    sim_matrix = torch.matmul(features, features.t()) / temperature  # [B, B]

    # 正样本: 相同标签
    labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
    # 负样本: 不同标签
    labels_ne = 1.0 - labels_eq

    # 排除自身
    mask = torch.eye(B, device=labels.device)
    labels_eq = labels_eq * (1.0 - mask)
    labels_ne = labels_ne * (1.0 - mask)

    # 对每个锚点，计算 InfoNCE
    loss = 0.0
    valid_count = 0
    for i in range(B):
        pos_mask = labels_eq[i].bool()
        neg_mask = labels_ne[i].bool()

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_sim = sim_matrix[i][pos_mask]
        neg_sim = sim_matrix[i][neg_mask]

        # 分母: 正样本 + 所有负样本
        denominator = torch.exp(pos_sim).sum() + torch.exp(neg_sim).sum()
        numerator = torch.exp(pos_sim).sum()

        if denominator > 0:
            loss = loss - torch.log(numerator / denominator)
            valid_count += 1

    if valid_count > 0:
        return loss / valid_count
    else:
        return torch.tensor(0.0, device=attn_weights.device)
