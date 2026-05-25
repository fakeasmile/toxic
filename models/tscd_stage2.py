import torch
from torch import nn


class ConceptDistiller(nn.Module):
    def __init__(self, input_dim, num_concepts, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
            nn.Sigmoid(),
        )

        self.num_concepts = num_concepts

    def forward(self, h):
        return self.probe(h)
