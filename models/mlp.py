import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, feature_vector):
        x = feature_vector
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x