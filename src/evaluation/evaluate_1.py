import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.datamodules import get_dataloaders
from src.models.architectures import DiabetesMLP
from src.models.hyperparametres import INPUT_DIM, HIDDEN_DIMS, BATCH_SIZE

def run_evaluation():
    # --- 1. CHEMINS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")
    model_path = os.path.join(project_root, "best_modele_diabete.pth")
    history_path = os.path.join(project_root, "historique_entrainement.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Démarrage du laboratoire d'évaluation...")

    # --- 2. TRACÉ DES COURBES (Depuis le JSON) ---
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(15, 6))
        
        # Graphique 1 : Loss (Train vs Val)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        plt.title('Évolution de la Loss (Diagnostic Overfitting)')
        plt.xlabel('Époques')
        plt.ylabel('Loss (BCE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Graphique 2 : ROC AUC (Train vs Val)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_roc_auc'], label='Train ROC AUC', color='blue', linewidth=2, linestyle='--')
        plt.plot(epochs, history['val_roc_auc'], label='Validation ROC AUC', color='purple', linewidth=2)
        plt.title('Évolution du ROC AUC')
        plt.xlabel('Époques')
        plt.ylabel('ROC AUC Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_path = os.path.join(project_root, "learning_curves_compare.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Courbes d'apprentissage sauvegardées sous : {plot_path}")
        
    except FileNotFoundError:
        print("Fichier historique introuvable. Lance train.py d'abord.")

    # --- 3. ÉVALUATION DU MEILLEUR MODÈLE (Depuis le .pth) ---
    _, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)
    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_y_true, all_y_probs = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            val_probs = torch.sigmoid(model(X_val))
            all_y_true.extend(y_val.cpu().numpy())
            all_y_probs.extend(val_probs.cpu().numpy())

    y_true = np.array(all_y_true).flatten()
    y_probs = np.array(all_y_probs).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    # Rapport de classification
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION (Seuil 0.5)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Sain (0)", "Diabétique (1)"]))

    # Matrice de Confusion
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Prédit Sain", "Prédit Diab"], 
                yticklabels=["Vrai Sain", "Vrai Diab"])
    plt.title("Matrice de Confusion (Seuil = 0.5)")
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')

    # Courbe ROC
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")

    eval_path = os.path.join(project_root, "evaluation_metrics.png")
    plt.savefig(eval_path, bbox_inches='tight')
    print(f"Matrice et Courbe ROC sauvegardées sous : {eval_path}")
    
    plt.show() # Affiche toutes les fenêtres de graphiques générées

if __name__ == "__main__":
    run_evaluation()