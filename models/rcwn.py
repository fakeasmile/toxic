import torch
from torch import nn
from transformers import AutoModel


class ConceptWhiteningLayer(nn.Module):
    def __init__(self, input_dim, num_concepts):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts

        self.weight = nn.Parameter(torch.randn(input_dim, num_concepts))
        nn.init.orthogonal_(self.weight)

    def orthogonalize(self):
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(self.weight.data, full_matrices=False)
            self.weight.data = U @ Vh

    def forward(self, h):
        W = self.weight
        z_c = h @ W
        z_r = h - z_c @ W.T
        return z_c, z_r

    def orthogonality_loss(self):
        W = self.weight
        gram = W.T @ W
        identity = torch.eye(self.num_concepts, device=W.device, dtype=W.dtype)
        return ((gram - identity) ** 2).sum()


class ConceptHead(nn.Module):
    def __init__(self, num_concepts, num_classes=2, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(num_concepts, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_c):
        return self.fc(self.dropout(z_c))


class ResidualHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_r):
        return self.net(z_r)


class RCWN(nn.Module):
    def __init__(self, plm_name, num_concepts, residual_hidden_dim=256,
                 num_classes=2, dropout=0.3, plm_frozen=True):
        super().__init__()

        self.plm = AutoModel.from_pretrained(plm_name)
        self.plm_hidden_size = self.plm.config.hidden_size

        if plm_frozen:
            for param in self.plm.parameters():
                param.requires_grad = False

        self.cw_layer = ConceptWhiteningLayer(self.plm_hidden_size, num_concepts)
        self.concept_scale = nn.Parameter(torch.ones(num_concepts))
        self.concept_shift = nn.Parameter(torch.zeros(num_concepts))
        self.concept_head = ConceptHead(num_concepts, num_classes, dropout)
        self.residual_head = ResidualHead(
            self.plm_hidden_size, residual_hidden_dim, num_classes, dropout
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, concept_targets=None):
        plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        h = plm_output.last_hidden_state[:, 0, :]

        z_c_raw, z_r = self.cw_layer(h)
        z_c = self.concept_scale * z_c_raw + self.concept_shift

        y_c = self.concept_head(z_c)
        y_r = self.residual_head(z_r)

        alpha = torch.sigmoid(self.alpha)
        y = alpha * y_c + (1 - alpha) * y_r

        losses = {}
        if concept_targets is not None:
            losses["align"] = ((z_c - concept_targets) ** 2).mean()

        losses["ortho"] = self.cw_layer.orthogonality_loss()

        return y, z_c, alpha, losses

    def get_concept_scores(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
            h = plm_output.last_hidden_state[:, 0, :]
            z_c_raw, _ = self.cw_layer(h)
            z_c = self.concept_scale * z_c_raw + self.concept_shift
        return z_c
