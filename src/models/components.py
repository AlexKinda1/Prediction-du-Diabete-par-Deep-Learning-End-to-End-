import torch.nn as nn


class SimpleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, x):
        return self.net(x)
