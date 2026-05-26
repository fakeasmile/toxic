import torch
import torch.nn as nn
from transformers import AutoModel


class LRDCBMModel(nn.Module):
    def __init__(
        self,
        plm_name: str = "chinese-roberta-wwm-ext",
        rationale_plm_name: str = "chinese-roberta-wwm-ext",
        num_concepts: int = 56,
        num_classes: int = 2,
        dropout: float = 0.3,
        share_plm: bool = True,
        concept_loss_weight: float = 0.1,
        use_rationale: bool = True,
    ):
        super().__init__()
        self.share_plm = share_plm
        self.use_rationale = use_rationale
        self.concept_loss_weight = concept_loss_weight

        self.text_plm = AutoModel.from_pretrained(plm_name)
        if not share_plm:
            self.rationale_plm = AutoModel.from_pretrained(rationale_plm_name)

        fusion_dim = 768 * 2 if use_rationale else 768

        self.dropout = nn.Dropout(dropout)
        self.concept_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, num_concepts),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        rationale_ids,
        rationale_mask,
        labels=None,
        concept_labels=None,
    ):
        text_embed = self.text_plm(input_ids, attention_mask).last_hidden_state[:, 0, :]

        if self.use_rationale:
            if self.share_plm:
                rationale_embed = self.text_plm(rationale_ids, rationale_mask).last_hidden_state[:, 0, :]
            else:
                rationale_embed = self.rationale_plm(rationale_ids, rationale_mask).last_hidden_state[:, 0, :]
            fused = torch.cat([text_embed, rationale_embed], dim=1)
        else:
            fused = text_embed

        fused = self.dropout(fused)
        concept_probs = self.concept_bottleneck(fused)
        logits = self.classifier(concept_probs)

        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            if concept_labels is not None:
                concept_loss = nn.MSELoss()(concept_probs, concept_labels)
                total_loss = ce_loss + self.concept_loss_weight * concept_loss
            else:
                total_loss = ce_loss
            return logits, concept_probs, total_loss

        return logits, concept_probs
