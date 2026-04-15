from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import joblib
import pandas as pd
import numpy as np
import os
import sys
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.architectures import DiabetesMLP

# ==========================================
# 1. CHARGEMENT EN MÉMOIRE (Lifespan)
# ==========================================
# Dictionnaire pour stocker proprement nos artefacts ML
ml_models = {}
OPTIMAL_THRESHOLD = 0.30 # À adapter avec ton meilleur seuil

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Démarrage du serveur : Chargement des artefacts ML en mémoire...")
    try:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))

        # Chemins absolus
        model_dir = os.path.join(project_root, "src/training/resultats_optuna_mlflow")
        modele_path = os.path.join(model_dir, "meilleur_modele_trial_7.pth") 
        
        scaler_dir = os.path.join(project_root, "Donnee_pretraite")
        scaler_path = os.path.join(scaler_dir, "scaler_diabete.pkl")
        colonnes_path = os.path.join(scaler_dir, "colonnes_entrainement.pkl")
        
        # 1. Chargement du Prétraitement (Scaler)
        ml_models["scaler"] = joblib.load(scaler_path)
        
        # 2. Chargement des colonnes d'entraînement strictes
        ml_models["training_columns"] = joblib.load(colonnes_path)
        
        # 3. Chargement de l'Architecture (Trial 7 : 64 -> 32 -> 16)
        model = DiabetesMLP(input_dim=37, hidden_dims=[64, 32, 16], dropout_rate=0.25)
        
        # 4. Chargement des Poids
        model.load_state_dict(torch.load(modele_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        ml_models["model"] = model
        
        print("✅ MLOps : Scaler, Colonnes et Modèle chargés avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        
    yield # L'API tourne ici
    
    print("🛑 Arrêt du serveur : Nettoyage de la mémoire...")
    ml_models.clear()

# ==========================================
# 2. INITIALISATION DE L'APPLICATION 
# ==========================================
app = FastAPI(
    title="API Dépistage Diabète",
    description="API MLOps pour prédire le risque de diabète à partir de symptômes cliniques.",
    version="2.0",
    lifespan=lifespan # On attache le gestionnaire de cycle de vie
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En production, on mettra l'URL exacte de ton site React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. DÉFINITION DES DONNÉES ATTENDUES
# ==========================================
class PatientData(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float

    class Config:
        json_schema_extra = {
            "example": {
                "HighBP": 1.0, "HighChol": 1.0, "CholCheck": 1.0, "BMI": 27.0, 
                "Smoker": 1.0, "Stroke": 0.0, "HeartDiseaseorAttack": 0.0, 
                "PhysActivity": 0.0, "Fruits": 1.0, "Veggies": 1.0, 
                "HvyAlcoholConsump": 0.0, "AnyHealthcare": 1.0, "NoDocbcCost": 0.0, 
                "GenHlth": 4.0, "MentHlth": 20.0, "PhysHlth": 20.0, "DiffWalk": 1.0, 
                "Sex": 0.0, "Age": 57.0, "Education": 4.0, "Income": 7.0
            }
        }

# ==========================================
# 4. LA ROUTE DE PRÉDICTION
# ==========================================
@app.post("/predict")
def predict_diabetes(patient: PatientData):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modèle non chargé en mémoire.")

    try:
        # 1. Récupération des données du médecin
        data_dict = patient.model_dump()
        df_patient = pd.DataFrame([data_dict])
        
        # 2. Application du One-Hot Encoding
        # IMPORTANT : get_dummies ignore les floats par défaut. 
        # On doit lui préciser quelles colonnes sont catégorielles !
        continuous_cols = ['BMI', 'PhysHlth', 'Age', 'MentHlth']
        categorical_cols = [col for col in df_patient.columns if col not in continuous_cols]
        
        # On force les catégories en float pour générer les noms avec ".0" (ex: Education_4.0)
        df_patient[categorical_cols] = df_patient[categorical_cols].astype(float)
        df_patient = pd.get_dummies(df_patient, columns=categorical_cols)
        
        # 3. ALIGNEMENT STRICT AVEC L'ENTRAÎNEMENT (Méthode très élégante)
        training_cols = ml_models["training_columns"]
        df_patient = df_patient.reindex(columns=training_cols, fill_value=0)
        
        # 4. FORÇAGE DU TYPE NUMÉRIQUE 
        df_patient = df_patient.astype(np.float32)
        
        # 5. PRÉTRAITEMENT (SCALER)
        # Attention: Ton scaler_diabete.pkl doit ABSOLUMENT avoir été entraîné sur ces 4 colonnes.
        # Si le script de prétraitement n'avait que ['BMI', 'PhysHlth', 'Age'], cela va crasher ici.
        df_patient[continuous_cols] = ml_models["scaler"].transform(df_patient[continuous_cols])
        
        # Vérification dans la console pour déboguer
        print("💡 COLONNES FINALES ENVOYÉES AU MODÈLE :", df_patient.columns.tolist())
        print("💡 VALEURS FINALES ENVOYÉES AU MODÈLE :", df_patient.values)
        
        # 6. Conversion en Tenseur PyTorch
        X_tensor = torch.FloatTensor(df_patient.values)
        
        # 7. Inférence du modèle
        with torch.no_grad():
            logit = ml_models["model"](X_tensor)
            probabilite = torch.sigmoid(logit).item()
            
        # 8. Application du Seuil Optimal
        is_diabetic = probabilite >= OPTIMAL_THRESHOLD
        
        return {
            "statut": "success",
            "diagnostic": "Risque Élevé (Diabétique)" if is_diabetic else "Risque Faible (Sain)",
            "probabilite_brute": round(probabilite, 4),
            "seuil_applique": OPTIMAL_THRESHOLD
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")