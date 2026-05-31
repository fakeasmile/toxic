import torch
from torch import nn


class KISCB(nn.Module):
    def __init__(self, plm_name, num_platforms=2, num_topics=4,
                 num_targets=5, num_strategies=4,
                 concept_emb_dim=128, dropout_rate=0.3):
        super(KISCB, self).__init__()

        from transformers import AutoModel
        self.plm = AutoModel.from_pretrained(plm_name)
        plm_hidden = self.plm.config.hidden_size

        self.platform_emb = nn.Embedding(num_platforms, 32)
        self.topic_emb = nn.Embedding(num_topics, 32)
        self.platform_topic_proj = nn.Linear(64, plm_hidden)

        self.target_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden, num_targets),
            nn.Sigmoid()
        )
        self.strategy_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden, num_strategies),
        )

        total_concepts = num_targets + num_strategies
        self.concept_proj = nn.Sequential(
            nn.Linear(total_concepts, concept_emb_dim),
            nn.ReLU()
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(plm_hidden, concept_emb_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(concept_emb_dim, 2)
        )

    def forward(self, input_ids, attention_mask, platform_ids, topic_ids):
        h = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        platform_feat = self.platform_emb(platform_ids)
        topic_feat = self.topic_emb(topic_ids)
        pt_feat = torch.cat([platform_feat, topic_feat], dim=-1)
        pt_gate = torch.sigmoid(self.platform_topic_proj(pt_feat))
        h = h * (1 + pt_gate)

        target_probs = self.target_head(h)
        strategy_logits = self.strategy_head(h)

        strategy_probs = torch.softmax(strategy_logits, dim=-1)

        concept_vec = torch.cat([target_probs, strategy_probs], dim=-1)
        c_emb = self.concept_proj(concept_vec)
        h_proj = self.residual_proj(h)
        fused = c_emb + h_proj
        logits = self.classifier(fused)

        return logits, target_probs, strategy_logits

    def get_explanation(self, target_probs, strategy_probs):
        target_names = ["种族相关", "性别相关", "地域相关", "LGBTQ相关", "无特定目标"]
        strategy_names = ["直接表达", "间接暗示", "反讽讽刺", "编码术语"]

        if isinstance(strategy_probs, torch.Tensor) and strategy_probs.dim() == 2:
            strategy_probs = torch.softmax(strategy_probs, dim=-1)

        explanation = {
            "target": {name: float(prob) for name, prob in zip(target_names, target_probs)},
            "strategy": {name: float(prob) for name, prob in zip(strategy_names, strategy_probs)},
        }
        return explanation
