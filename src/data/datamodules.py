import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(data, batch_size=32, shuffle=True):
    dataset = SimpleDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
