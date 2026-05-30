import torch
from torch import nn
from transformers import AutoModel


class ConceptTokenAttention(nn.Module):
    def __init__(self, concept_dim, hidden_dim, num_concepts, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.concept_embeddings)

        self.query_proj = nn.Linear(concept_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_repr, mask=None):
        """
        token_repr: [B, L, hidden_dim]
        mask: [B, L] (1=valid, 0=padding)
        """
        B, L, D = token_repr.shape
        N = self.num_concepts

        concept_queries = self.query_proj(self.concept_embeddings)  # [N, hidden_dim]
        keys = self.key_proj(token_repr)      # [B, L, hidden_dim]
        values = self.value_proj(token_repr)  # [B, L, hidden_dim]

        # 多头切分
        concept_q = concept_queries.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        k = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)                  # [B, H, L, d]
        v = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)                # [B, H, L, d]

        # 概念查询 token 键: [B, H, N, d] @ [B, H, d, L] -> [B, H, N, L]
        attn_scores = torch.matmul(concept_q.unsqueeze(0), k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, N, L]
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合: [B, H, N, L] @ [B, H, L, d] -> [B, H, N, d]
        concept_token_repr = torch.matmul(attn_weights, v)
        concept_token_repr = concept_token_repr.transpose(1, 2).reshape(B, N, D)
        concept_token_repr = self.out_proj(concept_token_repr)

        # 概念激活: 对 L 维度 max pool -> [B, N]
        concept_activations = attn_weights.max(dim=-1).values.mean(dim=1)  # [B, N]

        # Rationale 权重: 对概念维度求和 -> [B, L]
        rationale_weights = attn_weights.mean(dim=1).sum(dim=1)  # [B, L]

        return concept_token_repr, concept_activations, rationale_weights


class CGRM(nn.Module):
    def __init__(
        self,
        plm_name="chinese-roberta-wwm-ext",
        num_concepts=56,
        num_classes=2,
        concept_dim=64,
        hidden_dim=768,
        num_heads=4,
        dropout=0.3,
    ):
        super(CGRM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # PLM
        self.plm = AutoModel.from_pretrained(plm_name)

        # 概念-Token 交叉注意力
        self.concept_attn = ConceptTokenAttention(
            concept_dim, hidden_dim, num_concepts, num_heads, dropout
        )

        # 归一化 + Dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 分类层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels=None):
        # PLM 编码
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = outputs.last_hidden_state  # [B, L, 768]

        # 概念引导的 token 聚合
        concept_token_repr, concept_activations, rationale_weights = self.concept_attn(
            token_repr, mask=attention_mask
        )
        # concept_token_repr: [B, N, 768]
        # concept_activations: [B, N]
        # rationale_weights: [B, L]

        # 聚合: 对概念维度求和 -> [B, 768]
        aggregated = concept_token_repr.sum(dim=1)  # [B, 768]
        aggregated = self.norm(aggregated)
        aggregated = self.dropout(aggregated)

        # 分类
        x = self.fc1(aggregated)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        # 损失
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "concept_activations": concept_activations,
            "rationale_weights": rationale_weights,
        }

    def get_explanation(self, input_ids, attention_mask, tokenizer=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        concept_act = outputs["concept_activations"]  # [B, N]
        rationale_w = outputs["rationale_weights"]     # [B, L]

        topk_vals, topk_indices = torch.topk(concept_act, k=5, dim=-1)

        result = {
            "top_concepts": topk_indices.cpu().numpy(),
            "top_weights": topk_vals.cpu().numpy(),
            "all_concept_weights": concept_act.cpu().numpy(),
            "rationale_weights": rationale_w.cpu().numpy(),
        }

        if tokenizer is not None:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            result["tokens"] = tokens

        return result
