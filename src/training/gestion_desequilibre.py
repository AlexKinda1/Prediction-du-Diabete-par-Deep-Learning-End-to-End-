import pandas as pd

# Charger les données déjà prétraitées
df_train = pd.read_csv("Donnee_pretraite/diabetes_train_pretraite.csv")

# Vérifier le déséquilibre
print("Répartition des classes :")
print(df_train['Diabetes_binary'].value_counts())

print("\nProportions :")
print(df_train['Diabetes_binary'].value_counts(normalize=True))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Séparer X et y
X = df_train.drop("Diabetes_binary", axis=1)
y = df_train["Diabetes_binary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle avec gestion du déséquilibre
model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Résultats
print("\nConfusion Matrix :")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report :")
print(classification_report(y_test, y_pred))

# Probabilités
y_proba = model.predict_proba(X_test)[:, 1]

# Nouveau seuil (plus bas)
threshold = 0.3

y_pred_custom = (y_proba >= threshold).astype(int)

print("\nConfusion Matrix (seuil ajusté) :")
print(confusion_matrix(y_test, y_pred_custom))

print("\nClassification Report (seuil ajusté) :")
print(classification_report(y_test, y_pred_custom))

import shap

# Initialiser l'explainer
explainer = shap.Explainer(model, X_train)

# Calcul des valeurs SHAP
shap_values = explainer(X_test)

# Graphique global
shap.summary_plot(shap_values, X_test)