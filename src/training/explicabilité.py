import sys
import os

# Ajouter le dossier racine au path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.show()
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.models.architectures import DiabetesMLP

# Charger les données TEST 
df_test = pd.read_csv("Donnee_pretraite/diabetes_test_pretraite.csv")

# Séparer X et y
X_test = df_test.drop("Diabetes_binary", axis=1)
y_test = df_test["Diabetes_binary"]

# Convertir en numpy
X_np = X_test.values

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# Chemin vers le dossier contenant le modèle et ses paramètres
model_dir = os.path.join(project_root, "src/training/resultats_optuna_mlflow")

# Charger modèle
from src.models.architectures import DiabetesMLP

# Recréer le modèle
input_dim = X_np.shape[1]
model = DiabetesMLP(input_dim=input_dim)

# Charger les poids
model.load_state_dict(torch.load(os.path.join(model_dir, "meilleur_modele_trial_7.pth")))

# Mode évaluation
model.eval()

# Fonction compatible
def predict_proba_torch(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)

    probs = torch.sigmoid(outputs).numpy().flatten()
    probs = np.column_stack((1 - probs, probs))
    
    return probs

# LIME

# Créer explainer
explainer_lime = LimeTabularExplainer(
    X_np[:100],
    mode="classification",
    feature_names=X_test.columns,
    class_names=["Non diabétique", "Diabétique"],
    discretize_continuous=True
)

print("LIME démarre")

# Utiliser explainer
exp = explainer_lime.explain_instance(
    X_np[0],
    predict_proba_torch,
    num_features=10,
    num_samples=100
)

print("\n Explication LIME :")
print(exp.as_list())

# Sauvegarde
import os

print("Dossier actuel :", os.getcwd())

output_path = os.path.join(os.getcwd(), "lime.html")

exp.save_to_file(output_path)

print("Fichier LIME enregistré ici :", output_path)

print("LIME terminé")

# SHAP 

# On force une sortie 1D (classe positive uniquement)
def predict_shap(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)

    probs = torch.sigmoid(outputs).numpy().flatten()
    return probs   # une seule colonne

# Explainer
explainer = shap.KernelExplainer(predict_shap, X_np[:50])

# SHAP values
shap_values = explainer.shap_values(X_np[:100])

# Affichage (SANS [1])
shap.summary_plot(shap_values, X_np[:100], feature_names=X_test.columns)