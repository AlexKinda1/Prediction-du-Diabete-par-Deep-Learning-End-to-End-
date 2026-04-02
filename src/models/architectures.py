import torch
import torch.nn as nn

class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.2):
        super(DiabetesMLP, self).__init__()
        
        # On va stocker nos couches dans une liste PyTorch spéciale (ModuleList)
        layers = []
        
        # Dimension courante (commence avec la taille de l'entrée)
        current_dim = input_dim
        
        # CONSTRUCTION DYNAMIQUE DES COUCHES CACHÉES
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))   # 1. Multiplication Matricielle Wx + b
            layers.append(nn.BatchNorm1d(hidden_dim))           # 2. Stabilisation (Batch Normalization)
            layers.append(nn.ReLU())                            # 3. Non-linéarité avec la fonction d'activation ReLU
            layers.append(nn.Dropout(dropout_rate))             # 4. Régularisation
            
            # La dimension de sortie devient la dimension d'entrée de la prochaine couche
            current_dim = hidden_dim
            
        # On regroupe toutes ces couches en un seul bloc séquentiel
        self.feature_extractor = nn.Sequential(*layers)
        
        # --- COUCHE DE SORTIE ---
        # Un seul neurone en sortie car c'est une classification binaire.
        # Attention : Pas de Sigmoïde ici ! On retourne les "Logits" purs.
        self.classifier = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 1. Passage dans les couches cachées
        features = self.feature_extractor(x)
        
        # 2. Passage dans le neurone de décision final
        logits = self.classifier(features)
        
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