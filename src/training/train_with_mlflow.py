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

# Remplacement temporaire pour que le script soit autonome, 

from src.models.architectures import DiabetesMLP
from src.models.focal_loss import FocalLoss
from data.datamodules import get_dataloaders

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# INTEGRATION MLOPS & GREEN AI
# =====================================================================
import mlflow
import mlflow.pytorch
from codecarbon import EmissionsTracker

# =====================================================================
# CONFIGURATION DU DOSSIER DE RÉSULTATS LOCAL
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(script_dir, "resultats_optuna_mlflow")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f" Les résultats locaux seront sauvegardés dans : {RESULTS_DIR}")

# Configuration MLflow : Création de l'expérience globale
EXPERIMENT_NAME = "Optimisation_Diabete_MLP"
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================================================================
# FONCTION OBJECTIF OPTUNA (Avec Tracking MLflow)
# =====================================================================
def objective(trial):
    # Démarrage d'un "Run" MLflow pour cet essai spécifique
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        
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
        EPOCHS = 120  

        # ---> MLOPS : Enregistrement des hyperparamètres dans MLflow
        mlflow.log_params({
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "n_layers": n_layers,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        })

        # 2. Données et Modèle
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
        val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
        test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")

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
        best_optimal_threshold = 0.5 
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

            # ---> MLOPS : Enregistrement des métriques à chaque époque pour le suivi en direct
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_roc_auc": val_roc_auc
            }, step=epoch)

            # --- MISE À JOUR DU MEILLEUR MODÈLE ---
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_y_true = y_true_np
                
                # Calcul du seuil optimal avec l'Indice de Youden
                fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)
                youden_index = tpr - fpr 
                optimal_idx = np.argmax(youden_index)
                best_optimal_threshold = thresholds[optimal_idx]
                
                # Application du seuil optimal
                best_y_pred_classes = (y_probs_np >= best_optimal_threshold).astype(int)
                
                # Copie sécurisée sur CPU
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Sauvegarde des métriques internes pour Optuna
            trial.set_user_attr("roc_auc_history", val_roc_auc_history)
            trial.set_user_attr("optimal_threshold", float(best_optimal_threshold))

            # Élagage (Pruning)
            trial.report(val_roc_auc, epoch)
            if trial.should_prune():
                mlflow.set_tag("status", "pruned") # Indique dans MLflow que ce modèle a été arrêté tôt
                raise optuna.exceptions.TrialPruned()

        # =====================================================================
        # EXPORT DES ARTEFACTS VERS MLFLOW ET EN LOCAL
        # =====================================================================
        if best_model_state is not None:
            trial.set_user_attr("best_model_state", best_model_state)

        trial_num = trial.number
        
        # ---> MLOPS : Logging du meilleur seuil et du meilleur ROC AUC dans MLflow
        mlflow.log_metric("best_val_roc_auc", best_val_roc_auc)
        mlflow.log_metric("optimal_threshold", best_optimal_threshold)
        
        # 1. Graphique Courbes de Perte
        fig_loss = plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label="Train Loss", color='blue')
        plt.plot(val_loss_history, label="Validation Loss", color='orange')
        plt.title(f"Losses_Trial_{trial_num}")
        plt.xlabel("Époques")
        plt.ylabel("Focal Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(RESULTS_DIR, f"Losses_Trial_{trial_num}.png")) # Sauvegarde locale
        mlflow.log_figure(fig_loss, "courbes/loss_curve.png") # Sauvegarde MLflow
        plt.close(fig_loss)
        
        # 2. Graphique Matrice de Confusion 
        cm = confusion_matrix(best_y_true, best_y_pred_classes)
        fig_cm = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"CM_Trial_{trial_num} (Seuil: {best_optimal_threshold:.2f})")
        plt.xlabel("Prédictions")
        plt.ylabel("Valeurs Réelles")
        plt.savefig(os.path.join(RESULTS_DIR, f"CM_Trial_{trial_num}.png")) # Sauvegarde locale
        mlflow.log_figure(fig_cm, "matrices/confusion_matrix.png") # Sauvegarde MLflow
        plt.close(fig_cm)

        # 3. MLOPS : Enregistrement du modèle PyTorch complet dans MLflow
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            mlflow.pytorch.log_model(model, "modele_pytorch")

        return best_val_roc_auc

# =====================================================================
# BLOC PRINCIPAL ET RAPPORTS GLOBAUX
# =====================================================================
if __name__ == "__main__":
    print("Démarrage de l'optimisation des hyperparamètres avec Optuna et MLflow...")
    
    # ---> GREEN AI : Démarrage du tracker d'empreinte carbone
    tracker = EmissionsTracker(project_name="Diabete_MLP_Optuna")
    tracker.start()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25) 
    
    # ---> GREEN AI : Arrêt du tracker
    emissions_kg_co2 = tracker.stop()
    
# =====================================================================
    # INFOGRAPHIE GREEN AI (Empreinte Carbone)
    # =====================================================================
    km_voiture = emissions_kg_co2 / 0.192
    km_avion = emissions_kg_co2 / 0.250
    charges_tel = emissions_kg_co2 / 0.005
    
    # Création d'une figure au format paysage
    fig_green, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off') # On désactive les axes pour avoir un effet "Slide/Infographie"
    
    # Titre principal
    ax.text(0.5, 0.9, "🌱 RAPPORT GREEN AI - EMPREINTE CARBONE 🌱", 
            fontsize=20, fontweight='bold', ha='center', color='#27ae60')
    
    # Valeur Centrale (Le total)
    ax.text(0.5, 0.65, f"{emissions_kg_co2:.5f} kg CO₂eq", 
            fontsize=26, fontweight='bold', ha='center', color='#2c3e50', 
            bbox=dict(facecolor='#eaeded', edgecolor='#bdc3c7', boxstyle='round,pad=0.8'))
    
    ax.text(0.5, 0.52, "Émissions totales générées par l'entraînement complet (Optuna)", 
            fontsize=12, ha='center', color='#7f8c8d', style='italic')
    
    # Section des équivalences (Icônes et textes)
    # 1. Voiture
    ax.text(0.2, 0.30, "🚗", fontsize=50, ha='center')
    ax.text(0.2, 0.15, f"{km_voiture:.2f} km\nVoiture Thermique", 
            fontsize=13, ha='center', color='#34495e', fontweight='bold')
    
    # 2. Avion
    ax.text(0.5, 0.30, "✈️", fontsize=50, ha='center')
    ax.text(0.5, 0.15, f"{km_avion:.2f} km\nVol intérieur", 
            fontsize=13, ha='center', color='#34495e', fontweight='bold')
    
    # 3. Smartphone
    ax.text(0.8, 0.30, "📱", fontsize=50, ha='center')
    ax.text(0.8, 0.15, f"{charges_tel:.0f}\nRecharges complètes", 
            fontsize=13, ha='center', color='#34495e', fontweight='bold')
    
    # Signature en bas
    ax.text(0.5, 0.02, "Mesure effectuée via CodeCarbon", 
            fontsize=10, ha='center', color='#bdc3c7')
    
    plt.tight_layout()
    
    # Sauvegarde de l'infographie dans ton dossier local
    green_ai_path = os.path.join(RESULTS_DIR, "infographie_green_ai.png")
    plt.savefig(green_ai_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig_green)
    
    print(f"✅ Infographie Green AI générée avec succès : '{green_ai_path}'")
    
    # Graphique d'évolution du ROC_AUC pour TOUS les trials
    plt.figure(figsize=(12, 6)) 
    for trial in study.trials:
        if "roc_auc_history" in trial.user_attrs:
            history = trial.user_attrs["roc_auc_history"]
            plt.plot(history, alpha=0.6, linewidth=1.5, label=f"Trial {trial.number}")
            
    plt.title("Évolution du ROC_AUC pour l'ensemble des Trials")
    plt.xlabel("Époques")
    plt.ylabel("Validation ROC AUC")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout() 
    plt.savefig(os.path.join(RESULTS_DIR, "All_Trials_ROC_AUC.png"))
    plt.close()
    
    # Tableau récapitulatif
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.rename(columns={'value': 'Best_ROC_AUC'}, inplace=True)
    optimal_thresholds = [t.user_attrs.get('optimal_threshold', None) for t in study.trials]
    df_trials['Optimal_Threshold'] = optimal_thresholds
    csv_path = os.path.join(RESULTS_DIR, "recapitulatif_hyperparametres.csv")
    df_trials.to_csv(csv_path, index=False)
    
    # Sauvegarde locale du MEILLEUR modèle global
    best_trial = study.best_trial
    best_model_weights = best_trial.user_attrs.get("best_model_state")
    
    if best_model_weights is not None:
        model_filename = f"meilleur_modele_trial_{best_trial.number}.pth"
        model_filepath = os.path.join(RESULTS_DIR, model_filename)
        torch.save(best_model_weights, model_filepath)
        print(f"- Le MEILLEUR modèle global (Trial {best_trial.number}) a été sauvegardé localement sous : '{model_filepath}'")
        print("lancer 'mlflow ui' dans ton terminal pour voir le tableau de bord interactif !")

    print("\nAperçu des 5 meilleurs Trials :")
    top_5 = df_trials[df_trials['state'] == 'COMPLETE'].sort_values(by='Best_ROC_AUC', ascending=False).head(5)
    print(top_5.to_string(index=False))