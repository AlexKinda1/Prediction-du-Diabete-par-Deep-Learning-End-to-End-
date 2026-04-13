import sys
import os
import json
import optuna
import copy # Ajouté pour cloner l'état du modèle en mémoire

# Configuration des chemins
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from codecarbon import EmissionsTracker
from data.datamodules import get_dataloaders

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

from data.datamodules import get_dataloaders
# Assure-toi que l'import de ton architecture se fait correctement selon ton arborescence
# from src.models.architectures import DiabetesMLP

# =====================================================================
# CONFIGURATION DU DOSSIER DE RÉSULTATS (Chemins absolus)
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(script_dir, "resultats_optuna_1")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f" Les résultats seront sauvegardés dans : {RESULTS_DIR}")

# =====================================================================
# CLASSES DU MODÈLE ET LOSS 
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).float() 
        inputs = inputs.view(-1, 1)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

class DiabetesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(DiabetesMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

# =====================================================================
# FONCTION OBJECTIF OPTUNA
# =====================================================================
def objective(trial):
    # 1. Hyperparamètres
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    n_layers = trial.suggest_int("n_layers", 2, 3)
    hidden_dims = []
    in_features = 64
    for i in range(n_layers):
        hidden_dims.append(in_features)
        in_features = in_features // 2 
        
    INPUT_DIM = 37 
    BATCH_SIZE = 256
    EPOCHS = 150 

    # 2. Données et Modèle
# Dans CONFIG, remplace les chemins relatifs par des chemins absolus
# construits depuis la position du script lui-même




    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    train_path = os.path.join(project_root, "Donnee_pretraite/diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite/diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite/diabetes_test_pretraite.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)

    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Variables pour stocker l'historique du trial
    train_loss_history = []
    val_loss_history = []
    val_roc_auc_history = []
    
    # Variables pour stocker le MEILLEUR état du modèle
    best_val_roc_auc = 0.0
    best_y_true = []
    best_y_pred_classes = []
    best_optimal_threshold = 0.5 # Pour garder une trace du meilleur seuil
    best_model_state = None

    for epoch in range(EPOCHS):
        # --- ENTRAÎNEMENT ---
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad() 
            logits = model(X_batch)
            loss = criterion(logits, y_batch) 
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION ---
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
        val_loss_history.append(avg_val_loss)

        y_true_np = np.array(all_y_true).flatten()
        y_probs_np = np.array(all_y_probs).flatten()
        val_roc_auc = roc_auc_score(y_true_np, y_probs_np)
        val_roc_auc_history.append(val_roc_auc)

        # --- MISE À JOUR DU MEILLEUR MODÈLE ---
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_y_true = y_true_np
            
            # --- AJOUT: Calcul du seuil optimal avec l'Indice de Youden ---
            fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)
            youden_index = tpr - fpr # J = Sensibilité + Spécificité - 1  <=>  TPR - FPR
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            best_optimal_threshold = optimal_threshold
            
            # Application du seuil optimal calculé dynamiquement
            best_y_pred_classes = (y_probs_np >= optimal_threshold).astype(int)
            
            # On copie les poids du modèle vers le CPU pour éviter de saturer la VRAM (carte graphique)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Sauvegarde des métriques en temps réel pour Optuna
        trial.set_user_attr("roc_auc_history", val_roc_auc_history)
        
        # On peut aussi sauvegarder le seuil optimal dans les attributs d'Optuna si on veut le retrouver dans le dataframe
        trial.set_user_attr("optimal_threshold", float(best_optimal_threshold))

        # Élagage (Pruning)
        trial.report(val_roc_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # =====================================================================
    # EXPORT DES RÉSULTATS À LA FIN DU TRIAL (S'il n'a pas été élagué)
    # =====================================================================
    # On sauvegarde l'état du meilleur modèle dans ce trial pour pouvoir le récupérer plus tard
    if best_model_state is not None:
        trial.set_user_attr("best_model_state", best_model_state)

    trial_num = trial.number
    
    # 1. Classification Report (Basé sur la MEILLEURE époque et le SEUIL OPTIMAL)
    print(f"\n{'='*40}")
    print(f" CR_Trial_{trial_num} (Meilleure Époque | Seuil Optimal: {best_optimal_threshold:.4f})")
    print(f"{'='*40}")
    print(classification_report(best_y_true, best_y_pred_classes, zero_division=0))
    
    # 2. Graphique Courbes de Perte (Train vs Val)
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Train Loss", color='blue')
    plt.plot(val_loss_history, label="Validation Loss", color='orange')
    plt.title(f"Losses_Trial_{trial_num}")
    plt.xlabel("Époques")
    plt.ylabel("Focal Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, f"Losses_Trial_{trial_num}.png"))
    plt.close()
    
    # 3. Graphique Matrice de Confusion (Basée sur la MEILLEURE époque)
    cm = confusion_matrix(best_y_true, best_y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"CM_Trial_{trial_num} (Meilleure Époque | Seuil: {best_optimal_threshold:.2f})")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs Réelles")
    plt.savefig(os.path.join(RESULTS_DIR, f"CM_Trial_{trial_num}.png"))
    plt.close()

    return best_val_roc_auc

# =====================================================================
# BLOC PRINCIPAL ET RAPPORTS GLOBAUX
# =====================================================================
if __name__ == "__main__":
    print("Démarrage de l'optimisation des hyperparamètres avec Optuna...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25) 
    
    print("\n" + "="*50)
    print(" OPTIMISATION TERMINÉE - GÉNÉRATION DES RAPPORTS FINAUX")
    print("="*50)
    
    # 1. Graphique d'évolution du ROC_AUC pour TOUS les trials
    plt.figure(figsize=(12, 6)) # Agrandissement pour laisser de la place à la légende
    for trial in study.trials:
        if "roc_auc_history" in trial.user_attrs:
            history = trial.user_attrs["roc_auc_history"]
            # AJOUT: On nomme chaque ligne pour la légende
            plt.plot(history, alpha=0.6, linewidth=1.5, label=f"Trial {trial.number}")
            
    plt.title("Évolution du ROC_AUC pour l'ensemble des Trials")
    plt.xlabel("Époques")
    plt.ylabel("Validation ROC AUC")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # AJOUT: Placement de la légende à l'extérieur pour ne pas cacher les courbes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout() # Permet d'ajuster les marges pour que la légende rentre dans l'image
    
    plt.savefig(os.path.join(RESULTS_DIR, "All_Trials_ROC_AUC.png"))
    plt.close()
    print(f"- Graphique des courbes ROC_AUC globales sauvegardé dans '{RESULTS_DIR}/'.")

    # 2. Tableau récapitulatif
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.rename(columns={'value': 'Best_ROC_AUC'}, inplace=True)
    
    # Extraction du seuil optimal pour l'ajouter au dataframe final
    optimal_thresholds = [t.user_attrs.get('optimal_threshold', None) for t in study.trials]
    df_trials['Optimal_Threshold'] = optimal_thresholds
    
    csv_path = os.path.join(RESULTS_DIR, "recapitulatif_hyperparametres.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"- Tableau récapitulatif sauvegardé sous '{csv_path}'.")
    
    # 3. EXTRACTION ET SAUVEGARDE DU MEILLEUR MODÈLE GLOBAL (.pth)
    best_trial = study.best_trial
    best_model_weights = best_trial.user_attrs.get("best_model_state")
    
    if best_model_weights is not None:
        model_filename = f"meilleur_modele_trial_{best_trial.number}.pth"
        model_filepath = os.path.join(RESULTS_DIR, model_filename)
        torch.save(best_model_weights, model_filepath)
        print(f"- Le MEILLEUR modèle global (Trial {best_trial.number}) a été sauvegardé sous : '{model_filepath}'")
    else:
        print("-  Impossible de sauvegarder le modèle : les poids n'ont pas été trouvés en mémoire.")

    print("\nAperçu des 5 meilleurs Trials :")
    top_5 = df_trials[df_trials['state'] == 'COMPLETE'].sort_values(by='Best_ROC_AUC', ascending=False).head(5)
    print(top_5.to_string(index=False))
    
    
    