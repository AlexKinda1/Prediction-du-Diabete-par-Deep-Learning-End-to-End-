import sys
import os
# Pour que Python trouve le dossier src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import matplotlib.pyplot as plt
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Imports de ton projet
from src.models.architectures import DiabetesMLP
from data.datamodules import get_dataloaders



def evaluate_model():
    # 1. CONFIGURATION
    INPUT_DIM = 31
    HIDDEN_DIMS = [64, 32, 16]
    BATCH_SIZE = 64
    
    # Chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    project_root_2 = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Pour l'instant, on évalue sur la VALIDATION (Le test est pour le Sprint 3 final)
    train_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")
    val_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_val_pretraite.csv")
    test_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_test_pretraite.csv")
    model_path = os.path.join(project_root, "best_model_sprint3.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Début de l'évaluation sur : {device}")

    # 2. CHARGEMENT DES DONNÉES ET DU MODÈLE
    _, val_loader, _ = get_dataloaders(train_path, val_path, test_path, batch_size=BATCH_SIZE)
    
    model = DiabetesMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS).to(device)
    
    # On charge les poids sauvegardés
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Poids du modèle chargés avec succès !")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    # 3. PRÉDICTIONS
    model.eval()
    all_y_true = []
    all_y_probs = []

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            val_logits = model(X_val)
            val_probs = torch.sigmoid(val_logits)
            
            all_y_true.extend(y_val.cpu().numpy())
            all_y_probs.extend(val_probs.cpu().numpy())

    y_true = np.array(all_y_true).flatten()
    y_probs = np.array(all_y_probs).flatten()
    
    # Seuil par défaut à 0.5
    y_pred = (y_probs >= 0.5).astype(int)

    # 4. GÉNÉRATION DES RAPPORTS ET GRAPHIQUES
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    # target_names: 0 = Sain, 1 = Diabétique
    print(classification_report(y_true, y_pred, target_names=["Non_diabétique (0)", "Diabétique (1)"]))

    # 5. MATRICE DE CONFUSION (Seaborn)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Prédit Sain", "Prédit Diabétique"], 
                yticklabels=["Vrai Sain", "Vrai Diabétique"])
    plt.title("Matrice de Confusion (Seuil = 0.5)")
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction du Modèle')
    
    # Sauvegarde de l'image
    plot_path = os.path.join(project_root, "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"\nMatrice de confusion sauvegardée sous : {plot_path}")
    
    plt.show() # Affiche l'image si tu es dans un environnement qui le permet
    


def plot_learning_curves():
    # Remonte à la racine du projet pour trouver le fichier
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    history_path = os.path.join(project_root, "historique_sprint3.json")
    
    # 1. Chargement de l'historique
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(" Fichier historique introuvable. ")
        return

    # 2. Création de l'axe des X (le nombre d'époques)
    epochs = range(1, len(history['train_loss']) + 1)

    # 3. Paramétrage de la figure (2 graphiques côte à côte)
    plt.figure(figsize=(14, 5))

    # --- Graphique 1 : La fonction de coût (Loss) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Évolution de la Fonction de Coût (Loss)', fontsize=14)
    plt.xlabel('Époques (Epochs)', fontsize=12)
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- Graphique 2 : Les métriques de Performance ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_roc_auc'], label='Validation ROC AUC', color='purple', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='green', linewidth=2)
    plt.title('Évolution des Métriques', fontsize=14)
    plt.xlabel('Époques (Epochs)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Sauvegarde et affichage
    plot_path = os.path.join(project_root, "learning_curves.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Courbes d'apprentissage sauvegardées sous : {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    evaluate_model()
    plot_learning_curves()