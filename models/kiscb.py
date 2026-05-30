import torch
from torch import nn


class KISCB(nn.Module):
    def __init__(self, plm_name, num_coded_terms, num_platforms=2, num_topics=4,
                 num_targets=5, num_strategies=4, num_intents=5, num_tones=4,
                 concept_emb_dim=128, dropout_rate=0.3):
        super(KISCB, self).__init__()

        from transformers import AutoModel
        self.plm = AutoModel.from_pretrained(plm_name)
        plm_hidden = self.plm.config.hidden_size

        self.coded_term_proj = nn.Sequential(
            nn.Linear(num_coded_terms, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, plm_hidden * 2)
        )
        self.platform_emb = nn.Embedding(num_platforms, 32)
        self.topic_emb = nn.Embedding(num_topics, 32)
        self.platform_topic_proj = nn.Linear(64, plm_hidden * 2)

        self.target_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden * 2, num_targets),
            nn.Sigmoid()
        )
        self.strategy_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden * 2, num_strategies),
        )
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden * 2, num_intents),
            nn.Sigmoid()
        )
        self.tone_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(plm_hidden * 2, num_tones),
        )

        total_concepts = num_targets + num_strategies + num_intents + num_tones
        self.concept_proj = nn.Sequential(
            nn.Linear(total_concepts, concept_emb_dim),
            nn.ReLU()
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(plm_hidden * 2, concept_emb_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(concept_emb_dim, 2)
        )

    def forward(self, input_ids_orig, attention_mask_orig, input_ids_enhanced, attention_mask_enhanced,
                coded_term_features, platform_ids, topic_ids):
        h_orig = self.plm(input_ids=input_ids_orig, attention_mask=attention_mask_orig).last_hidden_state[:, 0, :]
        h_enhanced = self.plm(input_ids=input_ids_enhanced, attention_mask=attention_mask_enhanced).last_hidden_state[:, 0, :]
        h_fused = torch.cat([h_orig, h_enhanced], dim=-1)

        coded_term_injection = self.coded_term_proj(coded_term_features)
        h_fused = h_fused + coded_term_injection

        platform_feat = self.platform_emb(platform_ids)
        topic_feat = self.topic_emb(topic_ids)
        pt_feat = torch.cat([platform_feat, topic_feat], dim=-1)
        pt_gate = torch.sigmoid(self.platform_topic_proj(pt_feat))
        h_fused = h_fused * (1 + pt_gate)

        target_probs = self.target_head(h_fused)
        strategy_logits = self.strategy_head(h_fused)
        intent_probs = self.intent_head(h_fused)
        tone_logits = self.tone_head(h_fused)

        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        tone_probs = torch.softmax(tone_logits, dim=-1)

        concept_vec = torch.cat([target_probs, strategy_probs, intent_probs, tone_probs], dim=-1)
        c_emb = self.concept_proj(concept_vec)
        h_proj = self.residual_proj(h_fused)
        fused = c_emb + h_proj
        logits = self.classifier(fused)

        return logits, target_probs, strategy_logits, intent_probs, tone_logits

    def get_explanation(self, target_probs, strategy_probs, intent_probs, tone_probs):
        target_names = ["种族相关", "性别相关", "地域相关", "LGBTQ相关", "无特定目标"]
        strategy_names = ["直接表达", "间接暗示", "反讽讽刺", "编码术语"]
        intent_names = ["贬低", "歧视", "煽动", "物化", "无攻击意图"]
        tone_names = ["愤怒敌对", "蔑视轻蔑", "冷漠戏谑", "中性"]

        if isinstance(strategy_probs, torch.Tensor) and strategy_probs.dim() == 2:
            strategy_probs = torch.softmax(strategy_probs, dim=-1)
        if isinstance(tone_probs, torch.Tensor) and tone_probs.dim() == 2:
            tone_probs = torch.softmax(tone_probs, dim=-1)

        explanation = {
            "target": {name: float(prob) for name, prob in zip(target_names, target_probs)},
            "strategy": {name: float(prob) for name, prob in zip(strategy_names, strategy_probs)},
            "intent": {name: float(prob) for name, prob in zip(intent_names, intent_probs)},
            "tone": {name: float(prob) for name, prob in zip(tone_names, tone_probs)},
        }
        return explanation
