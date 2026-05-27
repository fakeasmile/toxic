import torch
import torch.nn as nn
from transformers import AutoModel


class CoTDCBMModel(nn.Module):
    def __init__(
        self,
        plm_name,
        num_concepts=56,
        num_classes=2,
        dropout=0.3,
        concept_loss_weight=0.1,
        soft_label_weight=0.5,
        use_residual=True,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        self.dropout = nn.Dropout(dropout)
        self.concept_layer = nn.Sequential(
            nn.Linear(768, num_concepts),
            nn.Sigmoid(),
        )
        self.use_residual = use_residual
        self.concept_loss_weight = concept_loss_weight
        self.soft_label_weight = soft_label_weight

        if use_residual:
            self.classifier = nn.Linear(768 + num_concepts, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, input_ids, attention_mask, labels=None, soft_labels=None, concept_labels=None):
        h = self.plm(input_ids, attention_mask).last_hidden_state[:, 0, :]
        h_drop = self.dropout(h)
        concept_probs = self.concept_layer(h_drop)

        if self.use_residual:
            fused = torch.cat([h_drop, concept_probs], dim=1)
        else:
            fused = concept_probs

        logits = self.classifier(fused)

        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss = ce_loss

            if soft_labels is not None:
                temperature = 2.0
                soft_loss = nn.KLDivLoss(reduction='batchmean')(
                    nn.functional.log_softmax(logits / temperature, dim=-1),
                    soft_labels / temperature,
                ) * (temperature ** 2)
                total_loss = total_loss + self.soft_label_weight * soft_loss

            if concept_labels is not None:
                concept_loss = nn.MSELoss()(concept_probs, concept_labels)
                total_loss = total_loss + self.concept_loss_weight * concept_loss

            return logits, concept_probs, total_loss

        return logits, concept_probs
