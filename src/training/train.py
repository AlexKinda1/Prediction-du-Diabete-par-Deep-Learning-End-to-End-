import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model, dataloader: DataLoader, optimizer, criterion, device='cpu'):
    model.train()
    for batch in dataloader:
        inputs = batch.to(device)
        logits = model(inputs)
        loss = criterion(logits, torch.zeros(len(inputs), dtype=torch.long, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    print('Lancement de la boucle d\'entraînement (stub)')
