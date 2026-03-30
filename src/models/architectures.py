import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(224*224*3, num_classes))

    def forward(self, x):
        return self.layers(x)
