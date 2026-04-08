import torch
from torch import nn

class FC(nn.Module):
    def __init__(self, dropout_rate):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, feature_vector):
        x = feature_vector
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x