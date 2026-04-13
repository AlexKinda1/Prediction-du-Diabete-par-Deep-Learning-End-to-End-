import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
    

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def prepare_and_split_data(raw_csv_path, output_dir):

    logger.info("1. Chargement du dataset nettoyé initial...")
    df = pd.read_csv(raw_csv_path) 
    df.columns = df.columns.str.strip()
    logger.info("csv lu avec succès.")

    # SUPPRESSION DES VARIABLES NON DÉSIRÉES
    
    cols_to_drop = ['Fruits', 'MentHlth', 'NoDocbcCost', 'Sex', 'AnyHealthcare', 'Veggies']
    df = df.drop(columns=cols_to_drop)
    
    logger.info("colonnes supprimées : " + ", ".join(cols_to_drop))

    # ONE-HOT ENCODING 
    logger.info("Application du One-Hot Encoding...")
    continuous_cols = ['BMI', 'PhysHlth', 'Age']
    categorical_cols = ['GenHlth', 'Education', 'Income'] 
    
    # get_dummies transforme automatiquement ces catégories en colonnes binaires
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    
    df.columns = df.columns.str.strip()

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
    logger.info(" Normalisation des variables continues (RobustScaler)")
    
    scaler = RobustScaler()
    # On FIT (apprend) et TRANSFORM sur le Train
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    
    # On applique seulement TRANSFORM sur Val et Test (pas de fit !)
    X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # RECONSTRUCTION ET SAUVEGARDE 
    logger.info("Sauvegarde des fichiers finaux.")
    # On recolle la cible (y) avec les features (X)
    train_final = pd.concat([X_train, y_train], axis=1)
    val_final = pd.concat([X_val, y_val], axis=1)
    test_final = pd.concat([X_test, y_test], axis=1)

    # Sauvegarde en CSV (index=False pour ne pas créer une colonne inutile)
    try:
        train_final.to_csv(f"{output_dir}/diabetes_train_pretraite.csv", index=False)
        val_final.to_csv(f"{output_dir}/diabetes_val_pretraite.csv", index=False)
        test_final.to_csv(f"{output_dir}/diabetes_test_pretraite.csv", index=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des fichiers: {e}")


if __name__ == "__main__":
    prepare_and_split_data("dataset.csv", "../Donnee_pretraite")
    
