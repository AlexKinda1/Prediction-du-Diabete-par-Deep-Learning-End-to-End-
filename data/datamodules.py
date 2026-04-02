import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        self.X = self.data.drop(columns=['Diabetes_binary']).values
        self.y = self.data['Diabetes_binary'].values
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1) # Le unsqueeze(1) transforme [N] en [N, 1] pour correspondre à la sortie du réseau


    def __len__(self):
        # Retourne le nombre total d'échantillons (patients)
        return len(self.data)

    def __getitem__(self, idx):
        # Retourne UN échantillon et son label à l'index 'idx'
        return self.X[idx], self.y[idx]

def get_dataloaders(train_path, val_path, test_path, batch_size: int = 64):

    # Instanciation des Datasets
    train_dataset = DiabetesDataset(train_path)
    val_dataset = DiabetesDataset(val_path)
    test_dataset = DiabetesDataset(test_path)
    
    # Création des DataLoaders
    # On mélange (shuffle=True) UNIQUEMENT le train set pour éviter que le modèle n'apprenne par cœur l'ordre des lignes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader