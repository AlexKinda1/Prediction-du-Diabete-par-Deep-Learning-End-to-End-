import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib # ---> AJOUT MLOPS : Import pour sauvegarder les objets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def prepare_and_split_data(raw_csv_path, output_dir):

    logger.info("1. Chargement du dataset nettoyé initial...")
    df = pd.read_csv(raw_csv_path) 
    df.columns = df.columns.str.strip()
    logger.info("csv lu avec succès.")

# ONE-HOT ENCODING 
    logger.info("Application du One-Hot Encoding...")
    continuous_cols = ['BMI', 'PhysHlth', 'Age', 'MentHlth', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'] # On garde ces 4 variables continues pour le scaler, le reste est catégoriel
    continuous_cols_for_scaler = ['BMI', 'PhysHlth', 'Age', 'MentHlth'] # Seules ces 4 variables sont continues et doivent être scalées, les autres sont catégorielles
    
    # ---> CORRECTION ICI : On exclut aussi la variable cible 'Diabetes_binary'
    cols_to_exclude = continuous_cols + ['Diabetes_binary']
    categorical_cols = df.columns.difference(cols_to_exclude) 
    
    # get_dummies transforme uniquement les variables explicatives catégorielles
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    
    df.columns = df.columns.str.strip()

    # Maintenant, 'Diabetes_binary' existe toujours !
    # SÉPARATION TEMPORAIRE POUR LE SPLIT
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']
     
    logger.info("Division des données (Stratifiée)")
    
    # On sépare d'abord 70% Train et 30% Reste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # On divise le Reste en deux pour avoir 15% Validation et 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # NORMALISATION SANS FUITE DE DONNÉES
    logger.info("Normalisation des variables continues (RobustScaler)")
    
    scaler = RobustScaler()
    # On FIT (apprend) et TRANSFORM sur le Train
    X_train[continuous_cols_for_scaler] = scaler.fit_transform(X_train[continuous_cols_for_scaler])
    
    # On applique seulement TRANSFORM sur Val et Test (pas de fit !)
    X_val[continuous_cols_for_scaler] = scaler.transform(X_val[continuous_cols_for_scaler])
    X_test[continuous_cols_for_scaler] = scaler.transform(X_test[continuous_cols_for_scaler])

    # ==========================================================
    # ---> AJOUTS MLOPS : SAUVEGARDE DES ARTEFACTS POUR L'API
    # ==========================================================
    logger.info("Sauvegarde des artefacts MLOps (Scaler et Colonnes)...")
    
    # 1. Sauvegarde du Scaler
    scaler_path = os.path.join(output_dir, "scaler_diabete.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 2. Sauvegarde de la liste exacte des 37 colonnes d'entraînement
    # C'est vital pour reconstruire le patient dans le bon ordre dans l'API
    cols_path = os.path.join(output_dir, "colonnes_entrainement.pkl")
    joblib.dump(list(X_train.columns), cols_path)
    # ==========================================================
    """
    # RECONSTRUCTION ET SAUVEGARDE 
    logger.info("Sauvegarde des fichiers finaux.")
    # On recolle la cible (y) avec les features (X)
    train_final = pd.concat([X_train, y_train], axis=1)
    val_final = pd.concat([X_val, y_val], axis=1)
    test_final = pd.concat([X_test, y_test], axis=1)

    # Sauvegarde en CSV (index=False pour ne pas créer une colonne inutile)
    try:
        train_final.to_csv(f"{output_dir}/diabetes_train_pretraite_1.csv", index=False)
        val_final.to_csv(f"{output_dir}/diabetes_val_pretraite_1.csv", index=False)
        test_final.to_csv(f"{output_dir}/diabetes_test_pretraite_1.csv", index=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des fichiers: {e}") """


if __name__ == "__main__":
    #print(pd.read_csv("dataset.csv").head())
    #df = pd.read_csv("../Donnee_pretraite/diabetes_train_pretraite_1.csv")
    #print(df.shape)
    prepare_and_split_data("dataset.csv", "../Donnee_pretraite")
    
    