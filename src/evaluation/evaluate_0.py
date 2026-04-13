"""
================================================================================
  SPRINT 3 — ÉVALUATION FINALE DU MODÈLE CHAMPION
================================================================================
 
  Ce script est organisé en 3 blocs indépendants :

  BLOC 1 — Évaluation standard sur le jeu de test
           Matrice de confusion, ROC, Précision-Rappel,
           distribution des probabilités, rapport comparatif

  BLOC 2 — Analyse des biais par sous-groupes
           Variables : Age, Sex, Income, Education, GenHlth
           Méthodes  : métriques classiques, Equalized Odds,
                       Disparate Impact, alertes par Z-test

  BLOC 3 — Empreinte carbone & temps d'exécution
           Mesure réelle de l'inférence (CodeCarbon)
           Estimation de l'entraînement Optuna
           Visualisation comparative avec la vie quotidienne
================================================================================
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from scipy import stats
from codecarbon import EmissionsTracker

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score
)

warnings.filterwarnings("ignore")

# ── Chemin vers les modules du projet ────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.architectures import DiabetesMLP
from data.datamodules import get_dataloaders


# ==============================================================================
# 0. CONFIGURATION GLOBALE
#    ➜ Seule section à modifier pour adapter le script à votre environnement
# ==============================================================================
# Dans CONFIG, remplace les chemins relatifs par des chemins absolus
# construits depuis la position du script lui-même

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

CONFIG = {
    "MODEL_PATH" : os.path.join(project_root,
                   "src/training/resultats_optuna_1/meilleur_modele_trial_17.pth"),
    "TRAIN_PATH" : os.path.join(project_root,
                   "Donnee_pretraite/diabetes_train_pretraite.csv"),
    "VAL_PATH"   : os.path.join(project_root,
                   "Donnee_pretraite/diabetes_val_pretraite.csv"),
    "TEST_PATH"  : os.path.join(project_root,
                   "Donnee_pretraite/diabetes_test_pretraite.csv"),
    # Mets à jour le numéro du trial champion

    "INPUT_DIM"   : 37,
    "HIDDEN_DIMS"       : [64, 32],   
    "DROPOUT_RATE"      : 0.4447,      
    "OPTIMAL_THRESHOLD" : 0.3611,    
    "RESULTS_DIR": "resultats_evaluation_finale",
    "BATCH_SIZE" : 256,

    # --- Paramètres pour l'estimation de l'empreinte carbone ----------------
    # Durée totale estimée des 25 trials Optuna (en heures)
    # Modifie cette valeur selon le temps réel de ton entraînement
    "OPTUNA_DURATION_H"  : 2.0,
    # Type de machine utilisée pour l'entraînement
    # "cpu" ou "gpu" — adapte selon ton matériel
    "TRAINING_DEVICE"    : "cpu",
    # Pays d'entraînement (influence le facteur d'émission électrique)
    # "FR" = France (énergie nucléaire, très bas), "US", "DE", etc.
    "COUNTRY_CODE"       : "FR",
}


# Facteurs d'émission électrique par pays (gCO₂/kWh)
# Source : Our World in Data / IEA 2023
EMISSION_FACTORS = {"FR": 56, "US": 386, "DE": 380, "GB": 233, "CN": 555}

# Puissance typique d'un CPU ou GPU pendant l'entraînement (en Watts)
DEVICE_POWER_W = {"cpu": 65, "gpu": 250}

# Références CO₂ pour la visualisation comparative (en grammes de CO₂eq)
# Sources : ADEME, Carbon Brief, Strubell et al. 2019 "Energy and Policy..."
CO2_REFERENCES = {
    "Entraînement GPT-3\n(OpenAI, 2020)"       : 552_000_000,
    "Vol Paris → New York\n(aller simple)"      : 1_000_000,
    "Voiture thermique\n(100 km)"               : 21_000,
    "Streaming vidéo HD\n(1 heure)"             : 36,
    "Email avec\npièce jointe"                  : 50,
    "Chargement\npage web"                      : 0.5,
}


# ==============================================================================
# 1. UTILITAIRES GÉNÉRAUX
# ==============================================================================

def setup(config: dict) -> str:
    """Crée le dossier de résultats et retourne son chemin."""
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    print(f"\n[INFO] Résultats → {config['RESULTS_DIR']}/")
    return config["RESULTS_DIR"]


def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    """
    Reconstruit l'architecture du Trial 11 et charge les poids sauvegardés.
    On n'a PAS besoin de réentraîner : Optuna a sauvegardé le meilleur état.
    """
    model = DiabetesMLP(
        input_dim    = config["INPUT_DIM"],
        hidden_dims  = config["HIDDEN_DIMS"],
        dropout_rate = config["DROPOUT_RATE"],
    ).to(device)

    if not os.path.exists(config["MODEL_PATH"]):
        raise FileNotFoundError(
            f" Modèle introuvable : {config['MODEL_PATH']}\n"
            "Vérifiez que le script Optuna a bien sauvegardé le .pth."
        )

    # map_location permet de charger un modèle entraîné sur GPU
    # même si on évalue sur CPU, et inversement
    model.load_state_dict(torch.load(config["MODEL_PATH"],
                                     map_location=device))
    model.eval()   # ← IMPORTANT : désactive Dropout pour l'inférence
    print(f"Modèle chargé : {config['MODEL_PATH']}")
    return model


def get_predictions(model, loader, device: torch.device,
                    tracker: EmissionsTracker = None):
    """
    Fait passer le jeu de test dans le modèle.
    Si un tracker CodeCarbon est fourni, la mesure d'émission est active.

    Retourne :
        y_true  (np.ndarray) : vraies étiquettes binaires
        y_probs (np.ndarray) : probabilités prédites entre 0 et 1
        duration_s (float)   : durée de l'inférence en secondes
    """
    all_y_true, all_y_probs = [], []

    if tracker:
        tracker.start()

    t0 = time.perf_counter()

    # torch.no_grad() : pas de calcul de gradient en dehors de l'entraînement
    # → économie de mémoire et de temps
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch    = X_batch.to(device)
            logits     = model(X_batch)
            # sigmoid convertit le logit brut en probabilité [0, 1]
            probs      = torch.sigmoid(logits)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_probs.extend(probs.cpu().numpy())

    duration_s = time.perf_counter() - t0

    if tracker:
        tracker.stop()

    return (np.array(all_y_true).flatten(),
            np.array(all_y_probs).flatten(),
            duration_s)


def reconstruct_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit les variables sociodémographiques originales
    à partir des colonnes encodées (standardisation / one-hot).

    Nécessaire car l'analyse des biais requiert des groupes discrets
    lisibles, pas des valeurs encodées.
    """
    result = pd.DataFrame(index=df.index)

    # --- Age : valeur standardisée → label de tranche ---
    # Les 13 valeurs standardisées correspondent aux 13 tranches BRFSS
    age_map = {
        -1.80: "18-24", -1.50: "25-29", -1.25: "30-34",
        -1.00: "35-39", -0.75: "40-44", -0.50: "45-49",
        -0.25: "50-54",  0.00: "55-59",  0.25: "60-64",
         0.50: "65-69",  0.75: "70-74",  1.00: "75-79",
         1.25: "80+",
    }
    result["Age"] = df["Age"].map(age_map)

    # --- Sex : 0 = Femmes, 1 = Hommes (encodage BRFSS conservé) ---
    result["Sex"] = df["Sex"].map({0.0: "Femmes", 1.0: "Hommes"})

    # --- Income : one-hot → groupe 1 à 8 ---
    # argmax retrouve quelle colonne one-hot vaut 1
    income_cols = [c for c in df.columns if c.startswith("Income_")]
    result["Income"] = (
        df[income_cols]
        .idxmax(axis=1)
        .str.extract(r"Income_(\d+)")[0]
        .astype(int)
        .map({
            1: "Revenu 1",  2: "Revenu 2",
            3: "Revenu 3", 4: "Revenu 4",
            5: "Revenu 5", 6: "Revenu 6",
            7: "Revenu 7", 8: "Revenu 8",
        })
    )

    # --- Education : one-hot → niveau 1 à 6 ---
    edu_cols = [c for c in df.columns if c.startswith("Education_")]
    result["Education"] = (
        df[edu_cols]
        .idxmax(axis=1)
        .str.extract(r"Education_(\d+)")[0]
        .astype(int)
        .map({
            1: "Niv.1\n(Jamais scolarisé)", 2: "Niv.2\n(Primaire)",
            3: "Niv.3\n(Collège)",           4: "Niv.4\n(Lycée)",
            5: "Niv.5\n(Bac+1/2)",           6: "Niv.6\n(Bac+3 et +)",
        })
    )

    # --- GenHlth : one-hot → état de santé général 1 (excellent) à 5 (mauvais) ---
    gen_cols = [c for c in df.columns if c.startswith("GenHlth_")]
    result["GenHlth"] = (
        df[gen_cols]
        .idxmax(axis=1)
        .str.extract(r"GenHlth_(\d+)")[0]
        .astype(int)
        .map({
            1: "1", 2: "2",
            3: "3",       4: "4",
            5: "5",
        })
    )

    return result


# ==============================================================================
# BLOC 1 — ÉVALUATION STANDARD SUR LE JEU DE TEST
# ==============================================================================

def bloc1_evaluation(y_true, y_probs, threshold, results_dir):
    """
    Génère tous les graphiques et rapports d'évaluation du modèle
    sur le jeu de test avec le seuil optimal de Youden.
    """
    print("\n" + "="*60)
    print("  BLOC 1 — ÉVALUATION SUR LE JEU DE TEST")
    print("="*60)

    y_pred    = (y_probs >= threshold).astype(int)
    y_pred_05 = (y_probs >= 0.5).astype(int)   # baseline pour comparaison

    # ── 1.1 Rapport comparatif texte ────────────────────────────────────────
    auc = roc_auc_score(y_true, y_probs)
    ap  = average_precision_score(y_true, y_probs)

    lines = [
        "="*65,
        " RAPPORT D'ÉVALUATION FINALE — MODÈLE CHAMPION (Trial 17)",
        "="*65,
        f"\n  ROC AUC (indépendant du seuil)  : {auc:.4f}",
        f"  Average Precision               : {ap:.4f}",
        f"  Seuil de décision utilisé       : {threshold:.4f} (Youden)\n",
        "-"*65,
        " Rapport de classification — seuil optimal",
        "-"*65,
        classification_report(y_true, y_pred,
                               target_names=["Sain (0)", "Diabétique (1)"],
                               zero_division=0),
        "-"*65,
        f" {'Métrique':<26} {'Seuil 0.5':>10} {'Seuil optimal':>14}",
        f" {'-'*26} {'-'*10} {'-'*14}",
        f" {'Accuracy':<26} {(y_pred_05==y_true).mean():>10.4f}"
        f" {(y_pred==y_true).mean():>14.4f}",
        f" {'Recall  Diabétiques':<26}"
        f" {recall_score(y_true,y_pred_05,pos_label=1,zero_division=0):>10.4f}"
        f" {recall_score(y_true,y_pred,pos_label=1,zero_division=0):>14.4f}",
        f" {'Précision Diabétiques':<26}"
        f" {precision_score(y_true,y_pred_05,pos_label=1,zero_division=0):>10.4f}"
        f" {precision_score(y_true,y_pred,pos_label=1,zero_division=0):>14.4f}",
        f" {'F1 Diabétiques':<26}"
        f" {f1_score(y_true,y_pred_05,pos_label=1,zero_division=0):>10.4f}"
        f" {f1_score(y_true,y_pred,pos_label=1,zero_division=0):>14.4f}",
        "="*65,
    ]
    report = "\n".join(lines)
    print(report)
    path = os.path.join(results_dir, "rapport_evaluation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Rapport texte → {path}")

    # ── 1.2 Figure principale : 4 graphiques en grille ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Évaluation du Modèle Champion — Jeu de Test",
                 fontsize=15, fontweight="bold", y=1.01)

    # --- Graphique A : Matrice de confusion ----------------------------------
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Prédit Sain", "Prédit Diabétique"],
                yticklabels=["Vrai Sain", "Vrai Diabétique"],
                annot_kws={"size": 13})
    ax.set_title(f"Matrice de Confusion\n(Seuil optimal = {threshold:.4f})",
                 fontsize=12)
    ax.set_ylabel("Réalité", fontsize=10)
    ax.set_xlabel("Prédiction", fontsize=10)

    # Annotations explicatives dans les cases
    labels = [["Vrais\nNégatifs", "Faux\nPositifs"],
              ["Faux\nNégatifs", "Vrais\nPositifs"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "gray")

    # --- Graphique B : Courbe ROC -------------------------------------------
    ax = axes[0, 1]
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    youden     = tpr - fpr
    opt_idx    = np.argmax(youden)

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"Courbe ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC = 0.5)")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color="red", zorder=5, s=120,
               label=f"Seuil Youden = {threshold:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.set_title(f"Courbe ROC — AUC = {auc:.4f}", fontsize=12)
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=10)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Graphique C : Courbe Précision-Rappel ------------------------------
    ax = axes[1, 0]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    baseline = y_true.mean()

    ax.plot(recall_curve, precision_curve, color="darkorange", lw=2,
            label=f"Courbe P-R (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline aléatoire ({baseline:.2f})")
    ax.fill_between(recall_curve, precision_curve, baseline,
                    alpha=0.08, color="darkorange")
    ax.set_title(f"Courbe Précision-Rappel — AP = {ap:.4f}", fontsize=12)
    ax.set_xlabel("Rappel (Recall)", fontsize=10)
    ax.set_ylabel("Précision", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Graphique D : Distribution des probabilités -------------------------
    ax = axes[1, 1]
    df_prob = pd.DataFrame({"prob": y_probs, "label": y_true})
    ax.hist(df_prob[df_prob["label"] == 0]["prob"], bins=60,
            alpha=0.55, color="steelblue", label="Sains (0)", density=True)
    ax.hist(df_prob[df_prob["label"] == 1]["prob"], bins=60,
            alpha=0.55, color="tomato",   label="Diabétiques (1)", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"Seuil = {threshold:.4f}")
    ax.set_title("Distribution des Probabilités Prédites", fontsize=12)
    ax.set_xlabel("P(Diabétique)", fontsize=10)
    ax.set_ylabel("Densité", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, "evaluation_complete.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Figure évaluation (4 graphiques) → {path}")

    return {"auc": auc, "ap": ap,
            "recall_opt": recall_score(y_true, y_pred, pos_label=1,
                                       zero_division=0)}


# ==============================================================================
# BLOC 2 — ANALYSE DES BIAIS
# ==============================================================================

def _compute_group_metrics(y_true, y_probs, y_pred, mask):
    """
    Calcule toutes les métriques pour un sous-groupe défini par un masque booléen.
    Retourne None si le groupe n'a pas assez de cas positifs (< 10) pour être fiable.
    """
    n_pos = y_true[mask].sum()
    if n_pos < 10:
        return None

    yt, yp_prob, yp = y_true[mask], y_probs[mask], y_pred[mask]

    # TPR = Recall = TP / (TP + FN)
    tpr = recall_score(yt, yp, pos_label=1, zero_division=0)
    # FPR = FP / (FP + TN)
    tn  = ((yp == 0) & (yt == 0)).sum()
    fp  = ((yp == 1) & (yt == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Taux de prédictions positives (pour le Disparate Impact)
    pos_rate = yp.mean()

    return {
        "n_total"     : int(mask.sum()),
        "n_diabetiques": int(n_pos),
        "pct_diab"    : round(100 * n_pos / mask.sum(), 1),
        "ROC_AUC"     : round(roc_auc_score(yt, yp_prob), 4),
        "Recall"      : round(tpr, 4),
        "Precision"   : round(precision_score(yt, yp, pos_label=1,
                                              zero_division=0), 4),
        "F1"          : round(f1_score(yt, yp, pos_label=1,
                                       zero_division=0), 4),
        "TPR"         : round(tpr, 4),
        "FPR"         : round(fpr, 4),
        "pos_rate"    : round(pos_rate, 4),
    }


def _ztest_proportion(p_group, n_group, p_global, n_global):
    """
    Z-test de proportions : teste si le Recall d'un sous-groupe est
    significativement différent du Recall global.

    H0 : p_group == p_global  (pas de différence)
    Si p-value < 0.05 : la différence est statistiquement significative
    → on déclenche une alerte.

    Paramètres :
        p_group  : Recall du sous-groupe (proportion)
        n_group  : nombre de vrais diabétiques dans le sous-groupe
        p_global : Recall global
        n_global : nombre total de vrais diabétiques
    """
    # Écart-type sous H0 (pooled)
    se = np.sqrt(p_global * (1 - p_global) * (1/n_group + 1/n_global))
    if se == 0:
        return 1.0   # pas de différence détectable
    z      = (p_group - p_global) / se
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))   # test bilatéral
    return pvalue


def _plot_classic_metrics(df_bias, var_name, global_recall,
                          global_auc, results_dir):
    """Graphique 1 : métriques classiques par sous-groupe (barres horizontales)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(df_bias)*0.7 + 2)))
    fig.suptitle(f"Métriques par sous-groupe — {var_name}",
                 fontsize=13, fontweight="bold")

    groups = df_bias["Sous-groupe"].astype(str)

    # Couleurs : rouge si alerte (Z-test significatif), sinon bleu/orange
    colors_recall = ["tomato" if a else "steelblue"
                     for a in df_bias["Alerte_Recall"]]
    colors_auc    = ["tomato" if a else "darkorange"
                     for a in df_bias["Alerte_Recall"]]

    # Recall
    ax = axes[0]
    bars = ax.barh(groups, df_bias["Recall"], color=colors_recall, alpha=0.8)
    ax.axvline(global_recall, color="black", linestyle="--", lw=1.5,
               label=f"Recall global = {global_recall:.2f}")
    ax.set_title("Recall (Sensibilité)")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    # Valeurs sur les barres
    for bar, val in zip(bars, df_bias["Recall"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    # ROC AUC
    ax = axes[1]
    bars = ax.barh(groups, df_bias["ROC_AUC"], color=colors_auc, alpha=0.8)
    ax.axvline(global_auc, color="black", linestyle="--", lw=1.5,
               label=f"AUC global = {global_auc:.2f}")
    ax.set_title("ROC AUC")
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.5, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, df_bias["ROC_AUC"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    # Légende alertes
    patch_alert  = mpatches.Patch(color="tomato",    label="Alerte (Z-test p<0.05)")
    patch_normal = mpatches.Patch(color="steelblue", label="Normal")
    fig.legend(handles=[patch_normal, patch_alert],
               loc="lower center", ncol=2, fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(results_dir, f"biais_{var_name}_metriques.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Métriques classiques → {path}")


def _plot_equalized_odds(df_bias, var_name, global_tpr, global_fpr,
                         results_dir):
    """
    Graphique 2 : Equalized Odds — scatter TPR vs FPR par sous-groupe.

    Un modèle parfaitement équitable aurait tous les points
    superposés sur le point global (croix rouge).
    Plus un point s'en éloigne, plus le modèle est inéquitable
    pour ce sous-groupe.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Couleur différente par sous-groupe pour lisibilité
    palette = plt.cm.tab20.colors
    groups  = df_bias["Sous-groupe"].astype(str).tolist()

    for i, (_, row) in enumerate(df_bias.iterrows()):
        color = palette[i % len(palette)]
        ax.scatter(row["FPR"], row["TPR"], color=color,
                   s=180, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(str(row["Sous-groupe"]),
                    (row["FPR"], row["TPR"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    # Point de référence global
    ax.scatter(global_fpr, global_tpr, color="red", marker="X",
               s=250, zorder=6, label=f"Global (TPR={global_tpr:.2f}, FPR={global_fpr:.2f})")

    # Zone d'équité : cercle de rayon 0.05 autour du point global
    circle = plt.Circle((global_fpr, global_tpr), 0.05,
                         color="red", fill=False, linestyle="--",
                         alpha=0.4, label="Zone d'équité (±0.05)")
    ax.add_patch(circle)

    ax.set_xlabel("FPR — Taux de Faux Positifs (patients sains mal classés)", fontsize=10)
    ax.set_ylabel("TPR — Taux de Vrais Positifs (Recall)", fontsize=10)
    ax.set_title(f"Equalized Odds — {var_name}\n"
                 "Un modèle équitable : tous les points proches du point global",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.02, min(1, df_bias["FPR"].max() + 0.1))
    ax.set_ylim(max(0, df_bias["TPR"].min() - 0.1), 1.02)

    plt.tight_layout()
    path = os.path.join(results_dir, f"biais_{var_name}_equalized_odds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Equalized Odds      → {path}")


def _plot_disparate_impact(df_bias, var_name, results_dir):
    """
    Graphique 3 : Disparate Impact par sous-groupe.

    DI = taux de prédictions positives du groupe
         ÷ taux du groupe de référence (le mieux traité)

    Zones :
        DI < 0.80 → sous-détection discriminatoire (rouge)   — règle des 4/5
        0.80 ≤ DI ≤ 1.20 → zone équitable (vert)
        DI > 1.20 → sur-détection (orange)
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_bias)*0.7 + 2)))

    groups = df_bias["Sous-groupe"].astype(str)
    di     = df_bias["DI"]

    # Couleur selon la zone
    colors = []
    for v in di:
        if v < 0.8:
            colors.append("tomato")
        elif v > 1.2:
            colors.append("darkorange")
        else:
            colors.append("mediumseagreen")

    bars = ax.barh(groups, di, color=colors, alpha=0.85, edgecolor="white")

    # Lignes de référence réglementaires
    ax.axvline(1.0, color="black",     linestyle="-",  lw=1.5,
               label="Référence (DI = 1.0)")
    ax.axvline(0.8, color="tomato",    linestyle="--", lw=1.5,
               label="Seuil min. 4/5 (0.80) — sous-détection")
    ax.axvline(1.2, color="darkorange",linestyle="--", lw=1.5,
               label="Seuil max.  (1.20) — sur-détection")

    # Zone équitable en vert transparent
    ax.axvspan(0.8, 1.2, alpha=0.08, color="green", label="Zone équitable")

    # Valeurs sur les barres
    for bar, val in zip(bars, di):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlabel("Disparate Impact (DI)", fontsize=10)
    ax.set_title(
        f"Disparate Impact — {var_name}\n"
        "Réf. = groupe le mieux traité par le modèle (DI = 1.0)\n"
        "DI < 0.80 : sous-détection discriminatoire (règle des 4/5)",
        fontsize=11
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(results_dir, f"biais_{var_name}_disparate_impact.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Disparate Impact    → {path}")


def bloc2_bias_analysis(y_true, y_probs, threshold,
                        df_test_raw, results_dir, global_metrics):
    """
    Analyse complète des biais pour chaque variable sociodémographique.
    Pour chaque variable : métriques classiques + Equalized Odds + Disparate Impact.
    Génère aussi un tableau d'alertes global.
    """
    print("\n" + "="*60)
    print("  BLOC 2 — ANALYSE DES BIAIS")
    print("="*60)

    y_pred = (y_probs >= threshold).astype(int)

    # Métriques globales de référence
    global_recall = global_metrics["recall_opt"]
    global_auc    = global_metrics["auc"]
    tn_g = ((y_pred==0) & (y_true==0)).sum()
    fp_g = ((y_pred==1) & (y_true==0)).sum()
    global_fpr    = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0
    global_tpr    = global_recall

    # Reconstruction des groupes depuis les colonnes encodées
    df_groups = reconstruct_groups(df_test_raw)

    variables = ["Age", "Sex", "Income", "Education", "GenHlth"]

    # Tableau d'alertes global (toutes variables confondues)
    all_alerts = []

    for var in variables:
        print(f"\n  ── Variable : {var} ──")
        col    = df_groups[var]
        groups = col.dropna().unique()

        rows = []
        for group in groups:
            mask = (col == group).values
            m    = _compute_group_metrics(y_true, y_probs, y_pred, mask)
            if m is None:
                continue
            # Z-test : le Recall de ce groupe est-il significativement différent ?
            pvalue = _ztest_proportion(
                p_group  = m["Recall"],
                n_group  = m["n_diabetiques"],
                p_global = global_recall,
                n_global = int(y_true.sum()),
            )
            m["Sous-groupe"]   = group
            m["pvalue_Recall"] = round(pvalue, 4)
            m["Alerte_Recall"] = (pvalue < 0.05) and (m["Recall"] < global_recall)
            rows.append(m)

        if not rows:
            print(f"  [SKIP] Pas assez de cas pour {var}")
            continue

        df_bias = pd.DataFrame(rows)

        # --- Calcul du Disparate Impact ---------------------------------------
        # Référence = groupe avec le pos_rate le plus élevé (le mieux traité)
        ref_pos_rate = df_bias["pos_rate"].max()
        ref_group    = df_bias.loc[df_bias["pos_rate"].idxmax(), "Sous-groupe"]
        df_bias["DI"] = (df_bias["pos_rate"] / ref_pos_rate).round(4)

        print(f"  Référence DI : {ref_group} "
              f"(pos_rate = {ref_pos_rate:.3f})")

        # Tri par Recall décroissant pour lisibilité des graphiques
        df_bias = df_bias.sort_values("Recall", ascending=True).reset_index(drop=True)

        # Affichage console
        cols_display = ["Sous-groupe", "n_total", "n_diabetiques",
                        "Recall", "ROC_AUC", "FPR", "DI", "pvalue_Recall", "Alerte_Recall"]
        print(df_bias[cols_display].to_string(index=False))

        # Sauvegarde CSV
        csv_path = os.path.join(results_dir, f"biais_{var}.csv")
        df_bias.to_csv(csv_path, index=False)

        # Graphiques
        _plot_classic_metrics(df_bias, var, global_recall,
                              global_auc, results_dir)
        _plot_equalized_odds(df_bias, var, global_tpr,
                             global_fpr, results_dir)
        _plot_disparate_impact(df_bias, var, results_dir)

        # Collecte des alertes pour le tableau global
        alerts = df_bias[df_bias["Alerte_Recall"] == True][
            ["Sous-groupe", "Recall", "DI", "pvalue_Recall"]
        ].copy()
        alerts.insert(0, "Variable", var)
        all_alerts.append(alerts)

    # --- Tableau d'alertes global --------------------------------------------
    print("\n" + "="*60)
    print("  TABLEAU D'ALERTES GLOBAL (Z-test p < 0.05)")
    print("="*60)
    if all_alerts:
        df_alerts = pd.concat(all_alerts, ignore_index=True)
        df_alerts["Recall_global"] = round(global_recall, 4)
        df_alerts["Ecart_points"]  = (
            (df_alerts["Recall"] - global_recall) * 100
        ).round(1)
        df_alerts["DI_alerte"] = df_alerts["DI"].apply(
            lambda x: "⚠ Sous-détection" if x < 0.8
            else ("⚠ Sur-détection" if x > 1.2 else "OK")
        )
        print(df_alerts.to_string(index=False))
        path = os.path.join(results_dir, "alertes_biais_global.csv")
        df_alerts.to_csv(path, index=False)
        print(f"\n[OK] Tableau d'alertes → {path}")
    else:
        print("  Aucune alerte statistiquement significative détectée.")


# ==============================================================================
# BLOC 3 — EMPREINTE CARBONE & TEMPS D'EXÉCUTION
# ==============================================================================

def bloc3_carbon(config, inference_emissions_kg,
                 inference_duration_s, total_duration_s, results_dir):
    """
    Calcule et visualise l'empreinte carbone du projet complet.

    Deux composantes :
    1. Inférence  → mesurée en temps réel par CodeCarbon (grammes)
    2. Entraînement → estimé à partir de la durée, puissance et facteur d'émission

    La visualisation compare le total à des références de vie quotidienne
    sur une échelle logarithmique.
    """
    print("\n" + "="*60)
    print("  BLOC 3 — EMPREINTE CARBONE & TEMPS D'EXÉCUTION")
    print("="*60)

    # ── 3.1 Calcul de l'empreinte d'entraînement estimée ────────────────────
    # Formule : E(kWh) = Puissance(W) × Durée(h) / 1000
    #           CO₂(g) = E(kWh) × facteur_émission(gCO₂/kWh)
    country      = config["COUNTRY_CODE"]
    power_w      = DEVICE_POWER_W[config["TRAINING_DEVICE"]]
    factor       = EMISSION_FACTORS.get(country, EMISSION_FACTORS["FR"])
    duration_h   = config["OPTUNA_DURATION_H"]

    energy_kwh        = power_w * duration_h / 1000
    training_co2_g    = energy_kwh * factor
    inference_co2_g   = inference_emissions_kg * 1000  # kg → g

    total_co2_g       = training_co2_g + inference_co2_g

    # ── 3.2 Rapport texte ────────────────────────────────────────────────────
    print(f"\n  Durée totale d'exécution du script  : "
          f"{total_duration_s:.1f} s ({total_duration_s/60:.1f} min)")
    print(f"  Durée d'inférence (jeu de test)     : "
          f"{inference_duration_s:.2f} s")
    print(f"\n  Empreinte inférence (CodeCarbon)    : "
          f"{inference_co2_g:.4f} g CO₂eq")
    print(f"  Empreinte entraînement (estimation) : "
          f"{training_co2_g:.2f} g CO₂eq")
    print(f"  ── Hypothèses entraînement ──────────────────────────")
    print(f"     Appareil     : {config['TRAINING_DEVICE'].upper()} "
          f"({power_w} W)")
    print(f"     Durée Optuna : {duration_h} h (25 trials)")
    print(f"     Pays         : {country} "
          f"(facteur = {factor} gCO₂/kWh)")
    print(f"  TOTAL CO₂ projet                    : "
          f"{total_co2_g:.2f} g CO₂eq")

    # ── 3.3 Visualisation comparative ────────────────────────────────────────
    references = dict(CO2_REFERENCES)
    references[f"Votre modèle\n(projet complet)"] = total_co2_g

    # Tri par valeur croissante pour la lisibilité
    sorted_refs = dict(sorted(references.items(), key=lambda x: x[1]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Empreinte Carbone du Modèle — Mise en Perspective",
                 fontsize=14, fontweight="bold")

    labels = list(sorted_refs.keys())
    values = list(sorted_refs.values())

    # Couleurs : rouge pour votre modèle, bleu pour les références
    colors = ["tomato" if "modèle" in l else "steelblue" for l in labels]

    # --- Graphique A : échelle logarithmique ---------------------------------
    ax = axes[0]
    bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("CO₂ équivalent (grammes) — Échelle logarithmique",
                  fontsize=10)
    ax.set_title("Comparaison (échelle log)\nNécessaire car les ordres de grandeur\n"
                 "varient de 0.5g à 552 millions g", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Valeurs sur les barres
    for bar, val in zip(bars, values):
        label_txt = (f"{val:.1f}g" if val < 1000
                     else f"{val/1000:.1f}kg" if val < 1_000_000
                     else f"{val/1_000_000:.0f}t")
        ax.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                label_txt, va="center", fontsize=8)

    # --- Graphique B : zoom sur le bas du spectre ----------------------------
    ax = axes[1]
    # On exclut les références très grandes pour le zoom
    zoom_threshold = total_co2_g * 100
    zoom_refs   = {k: v for k, v in sorted_refs.items() if v <= zoom_threshold}
    zoom_labels = list(zoom_refs.keys())
    zoom_values = list(zoom_refs.values())
    zoom_colors = ["tomato" if "modèle" in l else "steelblue"
                   for l in zoom_labels]

    bars2 = ax.barh(zoom_labels, zoom_values,
                    color=zoom_colors, alpha=0.8, edgecolor="white")
    ax.set_xlabel("CO₂ équivalent (grammes) — Échelle linéaire", fontsize=10)
    ax.set_title("Zoom : comparaisons proches\ndu modèle (échelle linéaire)",
                 fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars2, zoom_values):
        ax.text(val + max(zoom_values)*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.2f}g", va="center", fontsize=9)

    # Légende
    patch_model = mpatches.Patch(color="tomato",    label="Votre modèle")
    patch_ref   = mpatches.Patch(color="steelblue", label="Références")
    fig.legend(handles=[patch_model, patch_ref],
               loc="lower center", ncol=2, fontsize=10, framealpha=0.9)

    # Annotation méthodologique
    fig.text(0.5, 0.01,
             f"Sources : ADEME, Carbon Brief, Strubell et al. 2019  |  "
             f"Entraînement estimé ({config['TRAINING_DEVICE'].upper()}, "
             f"{duration_h}h, {country})  |  "
             f"Inférence mesurée par CodeCarbon",
             ha="center", fontsize=7, color="gray")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(results_dir, "empreinte_carbone.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Empreinte carbone → {path}")

    # Sauvegarde JSON des chiffres pour traçabilité
    carbon_report = {
        "inference_co2_g"  : round(inference_co2_g, 6),
        "inference_duration_s": round(inference_duration_s, 3),
        "training_co2_g_estimated": round(training_co2_g, 2),
        "total_co2_g"      : round(total_co2_g, 2),
        "total_duration_s" : round(total_duration_s, 1),
        "hypotheses"       : {
            "device"      : config["TRAINING_DEVICE"],
            "power_w"     : power_w,
            "duration_h"  : duration_h,
            "country"     : country,
            "factor_gco2_kwh": factor,
        }
    }
    json_path = os.path.join(results_dir, "empreinte_carbone.json")
    with open(json_path, "w") as f:
        json.dump(carbon_report, f, indent=2)
    print(f"[OK] Rapport carbone JSON → {json_path}")


# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

def main():
    script_start = time.perf_counter()

    print("\n" + "="*60)
    print("  ÉVALUATION FINALE — SPRINT 3")
    print("="*60)

    results_dir = setup(CONFIG)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    # ── Chargement du modèle et des données ──────────────────────────────────
    model = load_model(CONFIG, device)

    _, _, test_loader = get_dataloaders(
        CONFIG["TRAIN_PATH"], CONFIG["VAL_PATH"], CONFIG["TEST_PATH"],
        batch_size=CONFIG["BATCH_SIZE"]
    )
    print("[INFO] Données de test chargées.")

    # ── Prédictions avec mesure d'émissions CodeCarbon ───────────────────────
    print("[INFO] Inférence en cours (CodeCarbon actif)...")
    tracker = EmissionsTracker(
        project_name   = "diabetes_inference_sprint3",
        output_dir     = results_dir,
        log_level      = "error",   # silencieux dans la console
        save_to_file   = True,
    )
    y_true, y_probs, inference_duration_s = get_predictions(
        model, test_loader, device, tracker=tracker
    )

    # Lecture de l'émission mesurée par CodeCarbon (en kg CO₂eq)
    emissions_csv = os.path.join(results_dir, "emissions.csv")
    if os.path.exists(emissions_csv):
        df_emissions        = pd.read_csv(emissions_csv)
        inference_emissions = df_emissions["emissions"].iloc[-1]  # kg
    else:
        inference_emissions = 0.0   # fallback si CodeCarbon n'a pas sauvegardé
        print("[ATTENTION] Fichier emissions.csv introuvable — "
              "émission inférence mise à 0.")

    print(f"[INFO] Inférence terminée en {inference_duration_s:.2f}s")

    threshold = CONFIG["OPTIMAL_THRESHOLD"]

    # ── BLOC 1 : Évaluation standard ─────────────────────────────────────────
    global_metrics = bloc1_evaluation(y_true, y_probs, threshold, results_dir)

    # ── BLOC 2 : Analyse des biais ───────────────────────────────────────────
    # On charge le CSV de test pour reconstruire les groupes encodés
    df_test_raw = pd.read_csv(CONFIG["TEST_PATH"])
    bloc2_bias_analysis(y_true, y_probs, threshold,
                        df_test_raw, results_dir, global_metrics)

    # ── BLOC 3 : Empreinte carbone ───────────────────────────────────────────
    total_duration_s = time.perf_counter() - script_start
    bloc3_carbon(CONFIG, inference_emissions,
                 inference_duration_s, total_duration_s, results_dir)

    print("\n" + "="*60)
    print(f"  TERMINÉ — Durée totale : {total_duration_s:.1f}s")
    print(f"  Tous les résultats dans : {results_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
