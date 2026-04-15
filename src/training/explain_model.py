import sys
import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular

# Import de ton architecture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import DiabetesMLP

# ==========================================
# 1. CHARGEMENT DES DONNÉES ET DU MODÈLE
# ==========================================
print(" Chargement des artefacts...")
# Chemins (à adapter selon ton arborescence exacte)

# 1. Calcul dynamique de la racine du projet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# 2. Construction des chemins absolus
model_path = os.path.join(project_root, "src", "training", "resultats_optuna_mlflow", "meilleur_modele_trial_7.pth") 
scaler_path = os.path.join(project_root, "Donnee_pretraite", "scaler_diabete.pkl")
cols_path = os.path.join(project_root, "Donnee_pretraite", "colonnes_entrainement.pkl")
train_data_path = os.path.join(project_root, "Donnee_pretraite", "diabetes_train_pretraite.csv")


scaler = joblib.load(scaler_path)
training_columns = joblib.load(cols_path)

# Chargement du modèle (Hyperparamètres à adapter selon ton meilleur modèle)
model = DiabetesMLP(input_dim=37, hidden_dims=[64, 32, 16], dropout_rate=0.25)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

# Pour SHAP et LIME, on a besoin d'un échantillon du jeu d'entraînement (Background)
# On prend 100 patients au hasard pour ne pas exploser la RAM
df_train = pd.read_csv(train_data_path)
X_train_background = df_train.drop(columns=['Diabetes_binary']).sample(100, random_state=42).values

# ==========================================
# 2. PRÉPARATION DU PATIENT TEST
# ==========================================
# C'est le patient de notre test API (Risque potentiel)
patient_data = {
    "HighBP": 1.0, "HighChol": 1.0, "CholCheck": 1.0, "BMI": 32.0, 
    "Smoker": 1.0, "Stroke": 0.0, "HeartDiseaseorAttack": 0.0, 
    "PhysActivity": 0.0, "Fruits": 1.0, "Veggies": 1.0, 
    "HvyAlcoholConsump": 0.0, "AnyHealthcare": 1.0, "NoDocbcCost": 0.0, 
    "GenHlth": 4.0, "MentHlth": 15.0, "PhysHlth": 20.0, "DiffWalk": 1.0, 
    "Sex": 1.0, "Age": 10.0, "Education": 4.0, "Income": 5.0
}

df_patient = pd.DataFrame([patient_data])

# Prétraitement exact de l'API
continuous_cols = ['BMI', 'PhysHlth', 'Age', 'MentHlth']
categorical_cols = [col for col in df_patient.columns if col not in continuous_cols]
df_patient[categorical_cols] = df_patient[categorical_cols].astype(float)
df_patient = pd.get_dummies(df_patient, columns=categorical_cols)
df_patient = df_patient.reindex(columns=training_columns, fill_value=0).astype(np.float32)
df_patient[continuous_cols] = scaler.transform(df_patient[continuous_cols])

X_patient = df_patient.values
patient_tensor = torch.FloatTensor(X_patient)

with torch.no_grad():
    prob_brute = torch.sigmoid(model(patient_tensor)).item()
print(f"\n🩺 Probabilité prédite pour ce patient : {prob_brute * 100:.2f}%\n")

# ==========================================
# 3. EXPLICABILITÉ AVEC LIME
# ==========================================
print("Génération de l'explication LIME...")

# Fonction Wrapper pour LIME (Doit renvoyer les probas [Classe 0, Classe 1])
def predict_fn_lime(x_numpy):
    tensor = torch.FloatTensor(x_numpy)
    with torch.no_grad():
        probs_1 = torch.sigmoid(model(tensor)).numpy()
        probs_0 = 1 - probs_1
        return np.hstack((probs_0, probs_1))

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_background,
    feature_names=training_columns,
    class_names=['Sain', 'Diabétique'],
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(X_patient[0], predict_fn_lime, num_features=10)

# Sauvegarde du graphique LIME
fig_lime = lime_exp.as_pyplot_figure()
plt.title("LIME : Sensibilité Locale (Poids des variables)")
plt.tight_layout()
fig_lime.savefig("lime_explication_patient.png")
plt.close(fig_lime)
print("Graphique LIME sauvegardé sous 'lime_explication_patient.png'")


# ==========================================
# 4. EXPLICABILITÉ AVEC SHAP
# ==========================================
print(" Génération de l'explication SHAP...")

# Fonction Wrapper pour SHAP (Renvoie un array 1D des probabilités)
def predict_fn_shap(x_numpy):
    tensor = torch.FloatTensor(x_numpy)
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).numpy().flatten()

# On utilise KernelExplainer (très robuste pour tout type de modèle)
# shap.kmeans résume nos 100 patients en 10 profils pour accélérer le calcul
background_summary = shap.kmeans(X_train_background, 10)
shap_explainer = shap.KernelExplainer(predict_fn_shap, background_summary)

# Calcul des valeurs SHAP
shap_values = shap_explainer.shap_values(X_patient)

# Option 1 : Force Plot (Interactif en HTML)
shap.initjs() # Nécessaire pour générer le HTML
force_plot = shap.force_plot(
    shap_explainer.expected_value, 
    shap_values[0], 
    df_patient.iloc[0], 
    feature_names=training_columns
)
shap.save_html("shap_force_plot.html", force_plot)
print("Graphique SHAP interactif sauvegardé sous 'shap_force_plot.html'")

# Option 2 : Bar Plot (Image statique)
plt.figure(figsize=(10, 6))
# On utilise la valeur absolue des contributions pour voir qui "bouge" le plus le modèle
shap.summary_plot(shap_values, X_patient, feature_names=training_columns, plot_type="bar", show=False)
plt.title("SHAP : Top 10 des variables influentes pour ce patient")
plt.tight_layout()
plt.savefig("shap_bar_patient.png")
plt.close()
print("Graphique SHAP statique sauvegardé sous 'shap_bar_patient.png'")