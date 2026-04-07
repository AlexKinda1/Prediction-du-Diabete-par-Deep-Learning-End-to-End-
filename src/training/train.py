import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from codecarbon import EmissionsTracker
from data.datamodules import get_dataloaders
from src.models.architectures import DiabetesMLP, DiabetesRL

def train_model_base():
    # Configuration des hyperparamètres (sera mis plus tard dans le fichier hyperparametres.py)
    INPUT_DIM = 31           
    HIDDEN_DIMS = [64, 32, 16]
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 200
    EPOCHS = 100
    
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
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [], 
        'val_roc_auc': []
    }

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
             
            # Appronfondir cette partie 
            optimizer.zero_grad() # A chercher : Faut-il faire un zero_grad avant ou après le backward ? (ou les deux ?)
            logits = model(X_batch)
            loss = criterion(logits, y_batch) # BCEWithLogitsLoss mésure la distance entre les logits (sortie brute du réseau) et les vraies étiquettes 
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
        
        y_pred_classes = (y_probs_np >= 0.5).astype(int) # On utilise un seuil de 0.5 pour l'instant, mais on peut chercher à l'optimiser plus tard (ex: en regardant la courbe ROC)

        # Calculs
        val_roc_auc = roc_auc_score(y_true_np, y_probs_np) # Prend les probas !
        val_acc = accuracy_score(y_true_np, y_pred_classes)
        val_recall = recall_score(y_true_np, y_pred_classes, zero_division=0)
        val_f1 = f1_score(y_true_np, y_pred_classes, zero_division=0)

        #
        print(f"\n--- Bilan Epoch {epoch+1}/{EPOCHS} ---")
        print(f"Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")
        print(f"Val ROC AUC: {val_roc_auc:.4f}  (Notre métrique cible)")
        print(f"Val Acc    : {val_acc:.4f} | Val Recall : {val_recall:.4f} | Val F1 : {val_f1:.4f}")
        print("-" * 40)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_roc_auc'].append(val_roc_auc)

    print("\n Entraînement terminé ! Le moteur fonctionne parfaitement.")
    
    print("\n Sauvegarde du modèle et de l'historique...")
    torch.save(model.state_dict(), "modele_diabete_sprint2.pth")

    with open("historique_entrainement.json", "w") as f:
        json.dump(history, f)

    print("Modèle et Historique sauvegardés avec succès !")


if __name__ == "__main__":
    train_model_base()
    

"""
def train_model_advanced():
    # --- CONFIGURATION ---
    INPUT_DIM = 31           
    HIDDEN_DIMS = [62, 31, 16, 4, 16, 31, 62] # On teste une architecture plus profonde et plus large (en s'inspirant de la pyramide inversée)
    DROPOUT_RATE = 0.4 # On augmente un peu le dropout pour limiter l'overfitting
    LEARNING_RATE = 0.015
    BATCH_SIZE = 100
    EPOCHS = 200 # On peut réduire à 50, on a vu que l'overfitting arrive vite
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")
    model_save_path = os.path.join(project_root, "best_model_sprint3.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'entraînement optimisé sur : {device}")

    train_loader, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)

    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout_rate=0).to(device)
    # model = DiabetesRL(input_dim=INPUT_DIM).to(device)
    
    # OPTIMISATION 1 - POS_WEIGHT
    # Ratio = Nb Negatifs / Nb Positifs (environ 85% / 15% = ~5.6)
    weight_tensor = torch.tensor([5.6]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor) 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001) # Ajout de L2 regularization

    # OPTIMISATION 2 - VARIABLES POUR L'EARLY STOPPING
    best_val_roc_auc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_roc_auc': []}

# Démarrage du traqueur carbone
    tracker = EmissionsTracker(
        project_name="Diabetes_Model_Optimization", 
        output_dir=project_root,
        log_level="warning",
        measure_power_secs=15,  # On repasse à 15s (ou 10s) pour éviter que l'estimateur Windows ne plante
        tracking_mode="machine" # Force l'analyse globale si le processus spécifique est bloqué
    )
    tracker.start()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

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
        y_true_np = np.array(all_y_true).flatten()
        y_probs_np = np.array(all_y_probs).flatten()
        
        val_roc_auc = roc_auc_score(y_true_np, y_probs_np)
        val_acc = accuracy_score(y_true_np, (y_probs_np >= 0.38).astype(int))

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_roc_auc'].append(val_roc_auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val ROC AUC: {val_roc_auc:.4f}")

        #  OPTIMISATION 2 - SAUVEGARDE CONDITIONNELLE
        if val_roc_auc > best_val_roc_auc:
            print(f"   Nouveau record ROC AUC ({val_roc_auc:.4f} > {best_val_roc_auc:.4f}) ! Sauvegarde du modèle...")
            best_val_roc_auc = val_roc_auc
            torch.save(model.state_dict(), model_save_path)

    # Sauvegarde de l'historique
    with open(os.path.join(project_root, "historique_sprint3.json"), "w") as f:
        json.dump(history, f)
    print("\nntraînement optimisé terminé !")
    
    # Arrêt du traqueur et sauvegarde
    emissions: float = tracker.stop()
    print(f"\nBilan Carbone : {emissions * 1000:.5f} grammes de CO2 eq. émis.")
    print("Entraînement optimisé et audit environnemental terminés !")

if __name__ == "__main__":
    train_model_advanced()

"""