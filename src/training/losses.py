import torch.nn as nn


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    # placeholder
    return nn.CrossEntropyLoss()(inputs, targets)
