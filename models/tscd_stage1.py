import torch
from torch import nn
from transformers import AutoModel


class FGM:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Stage1Model(nn.Module):
    def __init__(self, plm_name, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()

        self.plm = AutoModel.from_pretrained(plm_name)
        self.plm_hidden_size = self.plm.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.plm_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.projector = nn.Sequential(
            nn.Linear(self.plm_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, input_ids, attention_mask):
        plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        h = plm_output.last_hidden_state[:, 0, :]

        logits = self.classifier(h)
        proj = self.projector(h)

        return logits, proj, h


def supervised_contrastive_loss(proj, labels, temperature=0.15):
    features = nn.functional.normalize(proj, dim=1)
    similarity = features @ features.T / temperature

    batch_size = labels.shape[0]
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    positive_mask = label_mask & diag_mask

    exp_sim = torch.exp(similarity) * diag_mask.float()
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    positive_counts = positive_mask.sum(dim=1).float()
    positive_counts = torch.clamp(positive_counts, min=1)

    loss_per_sample = (log_prob * positive_mask.float()).sum(dim=1) / positive_counts
    loss = -loss_per_sample.mean()

    return loss
