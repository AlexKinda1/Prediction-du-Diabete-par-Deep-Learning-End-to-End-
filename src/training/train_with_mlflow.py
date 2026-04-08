import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

from data.datamodules import get_dataloaders
from src.models.architectures import DiabetesMLP
from src.models.hyperparametres import INPUT_DIM, HIDDEN_DIMS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE

def train_model_base():
    # --- 1. CONFIGURATION ---
    EPOCHS = 200 
    PATIENCE = 20 # Nombre d'époques sans amélioration avant de stopper (Early Stopping)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")
    model_save_path = os.path.join(project_root, "best_modele_diabete.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'entraînement sur : {device}")

    train_loader, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)
    
    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_roc_auc': []}

    # Variables pour le Model Checkpointing et l'Early Stopping
    best_val_roc_auc = 0.0
    epochs_without_improvement = 0

    # --- 2. CONFIGURATION MLFLOW ---
    mlflow.set_experiment("Diabetes_Prediction_Baseline")
    
    with mlflow.start_run():
        # Enregistrement des hyperparamètres dans MLflow
        mlflow.log_params({
            "input_dim": INPUT_DIM,
            "hidden_dims": str(HIDDEN_DIMS),
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "patience": PATIENCE
        })

        # --- 3. BOUCLE D'ENTRAÎNEMENT ---
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
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)

            # --- BOUCLE DE VALIDATION ---
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
            y_pred_classes = (y_probs_np >= 0.5).astype(int)

            val_roc_auc = roc_auc_score(y_true_np, y_probs_np)
            val_acc = accuracy_score(y_true_np, y_pred_classes)
            val_recall = recall_score(y_true_np, y_pred_classes, zero_division=0)
            val_f1 = f1_score(y_true_np, y_pred_classes, zero_division=0)

            # Enregistrement des métriques dans MLflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_roc_auc": val_roc_auc,
                "val_acc": val_acc,
                "val_recall": val_recall
            }, step=epoch)

            print(f"\n--- Bilan Epoch {epoch+1}/{EPOCHS} ---")
            print(f"Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")
            print(f"Val ROC AUC: {val_roc_auc:.4f} | Val Recall : {val_recall:.4f}")
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_roc_auc'].append(val_roc_auc)

            # --- MODEL CHECKPOINTING & EARLY STOPPING ---
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"   Nouveau record ROC AUC ! Modèle sauvegardé.")
            else:
                epochs_without_improvement += 1
                print(f"   Pas d'amélioration depuis {epochs_without_improvement} époque(s).")

            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly Stopping déclenché à l'époque {epoch+1} ! Le modèle a cessé d'apprendre.")
                break

        print("\nEntraînement terminé !")

        # Sauvegarde de l'historique local
        with open(os.path.join(project_root, "historique_entrainement.json"), "w") as f:
            json.dump(history, f)

        # --- 4. ÉVALUATION FINALE (Sur le meilleur modèle) ---
        print("\nGénération des graphiques finaux avec le meilleur modèle...")
        
        # On recharge les meilleurs poids
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        final_y_true, final_y_probs = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                final_y_probs.extend(torch.sigmoid(model(X_val)).cpu().numpy())
                final_y_true.extend(y_val.cpu().numpy())
                
        final_y_true = np.array(final_y_true).flatten()
        final_y_probs = np.array(final_y_probs).flatten()
        final_y_pred = (final_y_probs >= 0.5).astype(int)

        # Rapport de classification
        print("\n" + "="*50)
        print("RAPPORT DE CLASSIFICATION (Seuil 0.5)")
        print("="*50)
        report = classification_report(final_y_true, final_y_pred, target_names=["Sain", "Diabétique"])
        print(report)

        # Matrice de Confusion
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(final_y_true, final_y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Prédit Sain", "Prédit Diab"], yticklabels=["Vrai Sain", "Vrai Diab"])
        plt.title("Matrice de Confusion (Seuil = 0.5)")
        cm_path = os.path.join(project_root, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path) # Sauvegarde dans MLflow
        plt.close()

        # Courbe ROC
        fpr, tpr, _ = roc_curve(final_y_true, final_y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='purple', label=f'ROC curve (AUC = {best_val_roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend()
        roc_path = os.path.join(project_root, "roc_curve.png")
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path) # Sauvegarde dans MLflow
        plt.close()

        # Sauvegarde du modèle final dans MLflow
        mlflow.pytorch.log_model(model, "meilleur_modele")
        print("Graphiques générés et sauvegardés localement et sur MLflow !")

if __name__ == "__main__":
    train_model_base()