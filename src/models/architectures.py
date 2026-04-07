import torch
import torch.nn as nn

class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.0):
        super(DiabetesMLP, self).__init__()
        
        # On va stocker nos couches dans une liste PyTorch spéciale
        layers = []
        
        # Dimension courante (commence avec la taille de l'entrée)
        current_dim = input_dim
        
        # CONSTRUCTION DYNAMIQUE DES COUCHES CACHÉES
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))   
            layers.append(nn.BatchNorm1d(hidden_dim))           # 2. Stabilisation (Batch Normalization)
            layers.append(nn.SELU())                            # 3. Non-linéarité avec la fonction d'activation ReLU
            layers.append(nn.Dropout(dropout_rate))             # 4. Régularisation (A chercher)
            
            # La dimension de sortie devient la dimension d'entrée de la prochaine couche
            current_dim = hidden_dim
            
        # On regroupe toutes ces couches en un seul bloc séquentiel
        self.feature_extractor = nn.Sequential(*layers)
        
        # COUCHE DE SORTIE
        self.classifier = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 1. Passage dans les couches cachées
        features = self.feature_extractor(x)
        
        # 2. Passage dans le neurone de décision final
        logits = self.classifier(features)
        
        return logits
    
class DiabetesRL(nn.Module):
    def __init__(self, input_dim: int):
        super(DiabetesRL, self).__init__()
        
        # COUCHE UNIQUE (Perceptron / Régression Logistique)
        # On relie directement les variables d'entrée à l'unique neurone de sortie
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Passage direct dans le neurone de décision
        logits = self.classifier(x)
        
        return logits
   
 
"""
NOMBRE_DE_FEATURES = 24 

# 1. On instancie notre modèle
model = DiabetesMLP(input_dim=NOMBRE_DE_FEATURES)
print(model) # Affiche l'architecture du réseau

# 2. On simule un "batch" (lot) de 64 patients qui entrent dans le réseau
# torch.randn crée un tenseur de nombres aléatoires de dimension [64, NOMBRE_DE_FEATURES]
dummy_patients = torch.randn(64, NOMBRE_DE_FEATURES)

# 3. On fait passer ces faux patients dans le réseau (Forward Pass)
predictions_brutes = model(dummy_patients)

# 4. On vérifie la forme de sortie
print(f"\nForme des données en entrée : {dummy_patients.shape}")
print(f"Forme des prédictions en sortie : {predictions_brutes.shape}") 
# Doit afficher [64, 1] (1 prédiction par patient)

"""