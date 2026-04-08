import sys
import os
import json
#Pour remonter à la base du projet et accéder aux modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from data.datamodules import get_dataloaders
from src.models.architectures import DiabetesMLP
from src.models.hyperparametres import INPUT_DIM, HIDDEN_DIMS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE

def train_model_base():
    # --- 1. CONFIGURATION ---
    EPOCHS = 200 
    PATIENCE = 20 # Early Stopping
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")
    
    model_save_path = os.path.join(project_root, "best_modele_diabete.pth")
    history_save_path = os.path.join(project_root, "historique_entrainement.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'entraînement sur : {device}")

    train_loader, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)
    
    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # Dictionnaire d'historique enrichi avec les métriques d'entraînement
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 
        'train_roc_auc': [], 'val_roc_auc': []
    }

    best_val_roc_auc = 0.0
    epochs_without_improvement = 0

    # --- 2. BOUCLE D'ENTRAÎNEMENT ---
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        # Pour calculer les métriques sur le train
        all_train_true, all_train_probs = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() 
            logits = model(X_batch)
            loss = criterion(logits, y_batch) 
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Stockage pour les métriques de Train (sans impacter les gradients)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_true.extend(y_batch.cpu().numpy())
            all_train_probs.extend(probs)
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Calcul des métriques Train
        y_train_true = np.array(all_train_true).flatten()
        y_train_probs = np.array(all_train_probs).flatten()
        train_roc_auc = roc_auc_score(y_train_true, y_train_probs)
        train_acc = accuracy_score(y_train_true, (y_train_probs >= 0.5).astype(int))

        # --- 3. BOUCLE DE VALIDATION ---
        model.eval()
        total_val_loss = 0.0
        all_y_true, all_y_probs = [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)
                total_val_loss += val_loss.item()
                
                val_probs = torch.sigmoid(val_logits)
                all_y_true.extend(y_val.cpu().numpy())
                all_y_probs.extend(val_probs.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)

        y_val_true = np.array(all_y_true).flatten()
        y_val_probs = np.array(all_y_probs).flatten()
        
        val_roc_auc = roc_auc_score(y_val_true, y_val_probs)
        val_acc = accuracy_score(y_val_true, (y_val_probs >= 0.5).astype(int))

        # --- 4. SAUVEGARDE DANS L'HISTORIQUE ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_roc_auc'].append(train_roc_auc)
        history['val_roc_auc'].append(val_roc_auc)

        print(f"\n--- Bilan Epoch {epoch+1}/{EPOCHS} ---")
        print(f"Loss    | Train : {avg_train_loss:.4f} | Val : {avg_val_loss:.4f}")
        print(f"ROC AUC | Train : {train_roc_auc:.4f} | Val : {val_roc_auc:.4f}")

        # --- 5. MODEL CHECKPOINTING & EARLY STOPPING ---
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"   Nouveau record ROC AUC ! Modèle sauvegardé.")
        else:
            epochs_without_improvement += 1
            print(f"   Pas d'amélioration depuis {epochs_without_improvement} époque(s).")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly Stopping déclenché à l'époque {epoch+1} ! Le modèle a cessé de généraliser.")
            break

    # --- 6. EXPORTATION DU JSON À LA FIN ---
    with open(history_save_path, "w") as f:
        json.dump(history, f)
        
    print("\nEntraînement terminé ! Le modèle .pth et l'historique .json ont été générés.")

if __name__ == "__main__":
    train_model_base()