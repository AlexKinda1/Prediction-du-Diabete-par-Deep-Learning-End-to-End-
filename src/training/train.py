import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score


from data.datamodules import get_dataloaders
from src.models.architectures import DiabetesMLP

def train_model_base():
    # Configuration des hyperparamètres (sera mis plus tard dans le fichier hyperparametres.py)
    INPUT_DIM = 31           
    HIDDEN_DIMS = [64, 32, 16]
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 150
    
    # Chemin vers les fichiers de données
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'entraînement sur : {device}")

    
    train_loader, val_loader, _ = get_dataloaders(
        train_path, val_path, test_path, batch_size=BATCH_SIZE
    )
    
    #Revoir en profondeur cette partie plus tard
    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Chercher d'autres optimizers plus tard (RMSProp, AdamW, etc.)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
             
            # Appronfondir cette partie 
            optimizer.zero_grad() # A chercher : Faut-il faire un zero_grad avant ou après le backward ? (ou les deux ?)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        
        # Listes pour stocker les vraies valeurs et les prédictions pour Scikit-Learn
        all_y_true = []
        all_y_probs = []

        with torch.no_grad(): # Pas de calcul de gradient ici
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                # 1. On récupère les logits
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)
                total_val_loss += val_loss.item()
                
                # 2. On transforme les logits en probabilités (Sigmoïde)
                val_probs = torch.sigmoid(val_logits)
                
                # 3. On stocke dans nos listes (on ramène sur CPU et on convertit en Numpy)
                all_y_true.extend(y_val.cpu().numpy())
                all_y_probs.extend(val_probs.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)

        # Calcul des métrics pour scikit-learn (on doit faire ça à la fin de l'époque une fois qu'on a toutes les prédictions)
        # On convertit les listes en tableaux Numpy 1D
        y_true_np = np.array(all_y_true).flatten()
        y_probs_np = np.array(all_y_probs).flatten()
        
        # Pour Accuracy, F1 et Recall, il faut des classes dures (0 ou 1)
        
        y_pred_classes = (y_probs_np >= 0.5).astype(int)

        # Calculs
        val_roc_auc = roc_auc_score(y_true_np, y_probs_np) # Prend les probas !
        val_acc = accuracy_score(y_true_np, y_pred_classes)
        val_recall = recall_score(y_true_np, y_pred_classes, zero_division=0)
        val_f1 = f1_score(y_true_np, y_pred_classes, zero_division=0)

        # --- 6. AFFICHAGE DES RÉSULTATS DE L'ÉPOQUE ---
        print(f"\n--- Bilan Epoch {epoch+1}/{EPOCHS} ---")
        print(f"Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")
        print(f"Val ROC AUC: {val_roc_auc:.4f}  (Notre métrique cible)")
        print(f"Val Acc    : {val_acc:.4f} | Val Recall : {val_recall:.4f} | Val F1 : {val_f1:.4f}")
        print("-" * 40)

    print("\n Entraînement terminé ! Le moteur fonctionne parfaitement.")

if __name__ == "__main__":
    train_model_base()