"""
  Ce script est organisé en 3 blocs indépendants :

  BLOC 1 — Évaluation standard sur le jeu de test
           Matrice de confusion, ROC, Précision-Rappel,
           distribution des probabilités, rapport comparatif seuil 0.5 vs optimal

  BLOC 2 — Analyse des biais par sous-groupes (version robuste)
           Variables et méthodes améliorées :
             • Groupes reconstruits dynamiquement via pd.cut / seuils adaptatifs
             • Statistiques rigoureuses (Z-test "Groupe vs Reste", Correction Bonferroni)
             • Equalized Odds robuste avec Intervalles de Confiance à 95%
             • Matrices de confusion cliniques par groupe
             • Disparate impact (sous-détection uniquement)

  BLOC 3 — Empreinte carbone & temps d'exécution
           Mesure réelle de l'inférence (CodeCarbon)
           Estimation de l'entraînement Optuna
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
import matplotlib.gridspec as gridspec
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
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

# ── Chemin vers les modules du projet ────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.architectures import DiabetesMLP
from data.datamodules import get_dataloaders


# ==============================================================================
# 0. CONFIGURATION GLOBALE
# ==============================================================================
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

CONFIG = {
    "MODEL_PATH"        : os.path.join(project_root, "src/training/resultats_optuna_1/meilleur_modele_trial_17.pth"),
    "INPUT_DIM"         : 37,
    "HIDDEN_DIMS"       : [64, 32],
    "DROPOUT_RATE"      : 0.3353,
    "OPTIMAL_THRESHOLD" : 0.3413,

    "TRAIN_PATH" : os.path.join(project_root, "Donnee_pretraite/diabetes_train_pretraite.csv"),
    "VAL_PATH"   : os.path.join(project_root, "Donnee_pretraite/diabetes_val_pretraite.csv"),
    "TEST_PATH"  : os.path.join(project_root, "Donnee_pretraite/diabetes_test_pretraite.csv"),

    "RESULTS_DIR"        : "resultats_evaluation_v2_robuste",
    "BATCH_SIZE"         : 256,

    "OPTUNA_DURATION_H"  : 2.0,
    "TRAINING_DEVICE"    : "cpu",
    "COUNTRY_CODE"       : "FR",
}

EMISSION_FACTORS = {"FR": 56, "US": 386}
DEVICE_POWER_W = {"cpu": 65, "gpu": 250}

CO2_REFERENCES = {
    "Entraînement GPT-3\n(OpenAI, 2020)"  : 552_000_000,
    "Vol Paris → New York\n(aller simple)" : 1_000_000,
    "Voiture thermique\n(100 km)"          : 21_000,
    "Email avec\npièce jointe"             : 50,
    "Streaming vidéo HD\n(1 heure)"        : 36,
    "Chargement\npage web"                 : 0.5,
}

# Mapping constants pour les variables one-hot
INCOME_GROUP_MAP = {1: "Faibles revenus", 2: "Faibles revenus", 3: "Faibles revenus", 
                    4: "Revenus moyens", 5: "Revenus moyens", 6: "Revenus moyens", 
                    7: "Revenus élevés", 8: "Revenus élevés"}
EDUCATION_GROUP_MAP = {1: "(Niv.1-4)", 2: "(Niv.1-4)", 3: "(Niv.1-4)", 4: "(Niv.1-4)", 5: "(Niv.5-6)", 6: "(Niv.5-6)"}
GENHLTH_LABELS = {1: "(1)", 2: "(2)", 3: "(3)", 4: "(4)", 5: "(5)"}


# ==============================================================================
# 1. UTILITAIRES GÉNÉRAUX & RECONSTRUCTION DE DONNÉES
# ==============================================================================

def setup(config: dict) -> str:
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    print(f"\n[INFO] Résultats → {config['RESULTS_DIR']}/")
    return config["RESULTS_DIR"]

def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    model = DiabetesMLP(
        input_dim    = config["INPUT_DIM"],
        hidden_dims  = config["HIDDEN_DIMS"],
        dropout_rate = config["DROPOUT_RATE"],
    ).to(device)

    if not os.path.exists(config["MODEL_PATH"]):
        raise FileNotFoundError(f"[ERREUR] Modèle introuvable : {config['MODEL_PATH']}")

    model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=device))
    model.eval()
    print(f"[INFO] Modèle chargé : {config['MODEL_PATH']}")
    return model

def get_predictions(model, loader, device: torch.device, tracker: EmissionsTracker = None):
    all_y_true, all_y_probs = [], []
    if tracker: tracker.start()
    t0 = time.perf_counter()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            probs   = torch.sigmoid(logits)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_probs.extend(probs.cpu().numpy())

    duration_s = time.perf_counter() - t0
    if tracker: tracker.stop()

    return np.array(all_y_true).flatten(), np.array(all_y_probs).flatten(), duration_s


def reconstruct_groups_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit les groupes sociodémographiques de manière robuste.
    Ne dépend plus de valeurs exactes codées en dur pour l'âge et le sexe.
    """
    result = pd.DataFrame(index=df.index)

    # Age : Découpage par intervalles (bins). 
    # NOTE: Si "Age" est standardisé, ajustez les limites de 'bins' selon votre standardisation.
    # Ici, nous utilisons des seuils arbitraires illustratifs (-0.6, 0.3) pour du StandardScaler.
    result["Age"] = pd.cut(
        df["Age"], 
        bins=[-np.inf, -0.6, 0.3, np.inf], 
        labels=["Jeunes adultes\n(18-44 ans)", "Adultes\n(45-64 ans)", "Seniors\n(65 ans et +)"]
    )

    # Sex : Seuil < 0.5 pour éviter les erreurs d'arrondi des floats
    result["Sex"] = np.where(df["Sex"] < 0.5, "Femmes", "Hommes")

    # Income, Education, GenHlth : Extraction depuis les colonnes One-Hot (Robuste par nature)
    income_cols = [c for c in df.columns if c.startswith("Income_")]
    if income_cols:
        result["Income"] = df[income_cols].idxmax(axis=1).str.extract(r"Income_(\d+)")[0].astype(int).map(INCOME_GROUP_MAP)

    edu_cols = [c for c in df.columns if c.startswith("Education_")]
    if edu_cols:
        result["Education"] = df[edu_cols].idxmax(axis=1).str.extract(r"Education_(\d+)")[0].astype(int).map(EDUCATION_GROUP_MAP)

    gen_cols = [c for c in df.columns if c.startswith("GenHlth_")]
    if gen_cols:
        result["GenHlth"] = df[gen_cols].idxmax(axis=1).str.extract(r"GenHlth_(\d+)")[0].astype(int).map(GENHLTH_LABELS)

    return result


# ==============================================================================
# BLOC 1 — ÉVALUATION STANDARD SUR LE JEU DE TEST
# ==============================================================================
# [NOTE] Le Bloc 1 reste identique à votre code original (parfait pour l'évaluation globale)
def bloc1_evaluation(y_true, y_probs, threshold, results_dir):
    print("\n" + "="*60)
    print("  BLOC 1 — ÉVALUATION SUR LE JEU DE TEST")
    print("="*60)

    y_pred    = (y_probs >= threshold).astype(int)
    y_pred_02 = (y_probs >= 0.2).astype(int)

    auc = roc_auc_score(y_true, y_probs)
    ap  = average_precision_score(y_true, y_probs)

    lines = [
        "=" * 65,
        " RAPPORT D'ÉVALUATION FINALE — MODÈLE CHAMPION",
        "=" * 65,
        f"\n  ROC AUC (indépendant du seuil)  : {auc:.4f}",
        f"  Average Precision               : {ap:.4f}",
        f"  Seuil de décision utilisé       : {threshold:.4f} (Youden)\n",
        "-" * 65,
        " Rapport de classification — seuil optimal",
        "-" * 65,
        classification_report(y_true, y_pred, target_names=["Sain (0)", "Diabétique (1)"], zero_division=0),
        "-" * 65,
        f" {'Métrique':<26} {'Seuil 0.5':>10} {'Seuil optimal':>14}",
        f" {'-'*26} {'-'*10} {'-'*14}",
        f" {'Accuracy':<26}{(y_pred_02 == y_true).mean():>10.4f}{(y_pred == y_true).mean():>14.4f}",
        f" {'Recall Diabétiques':<26}{recall_score(y_true, y_pred_02, pos_label=1, zero_division=0):>10.4f}{recall_score(y_true, y_pred, pos_label=1, zero_division=0):>14.4f}",
        f" {'Précision Diabétiques':<26}{precision_score(y_true, y_pred_02, pos_label=1, zero_division=0):>10.4f}{precision_score(y_true, y_pred, pos_label=1, zero_division=0):>14.4f}",
        f" {'F1 Diabétiques':<26}{f1_score(y_true, y_pred_02, pos_label=1, zero_division=0):>10.4f}{f1_score(y_true, y_pred, pos_label=1, zero_division=0):>14.4f}",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(results_dir, "rapport_evaluation.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # 1.2 Graphiques en grille...
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Évaluation du Modèle Champion — Jeu de Test", fontsize=15, fontweight="bold", y=1.01)

    # Matrice
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0,0],
                xticklabels=["Prédit Sain", "Prédit Diab."], yticklabels=["Vrai Sain", "Vrai Diab."])
    axes[0,0].set_title(f"Matrice de Confusion\n(Seuil optimal = {threshold:.4f})")
    
    # ROC
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    axes[0,1].plot(fpr_arr, tpr_arr, color="steelblue", lw=2, label=f"Courbe ROC (AUC = {auc:.4f})")
    axes[0,1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0,1].fill_between(fpr_arr, tpr_arr, alpha=0.08, color="steelblue")
    axes[0,1].set_title("Courbe ROC")
    axes[0,1].legend()

    # P-R
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
    axes[1,0].plot(rec_arr, prec_arr, color="darkorange", lw=2, label=f"Courbe P-R (AP = {ap:.4f})")
    axes[1,0].fill_between(rec_arr, prec_arr, y_true.mean(), alpha=0.08, color="darkorange")
    axes[1,0].set_title("Courbe Précision-Rappel")
    axes[1,0].legend()

    # Dist
    df_prob = pd.DataFrame({"prob": y_probs, "label": y_true})
    axes[1,1].hist(df_prob[df_prob["label"] == 0]["prob"], bins=60, alpha=0.55, color="steelblue", label="Sains", density=True)
    axes[1,1].hist(df_prob[df_prob["label"] == 1]["prob"], bins=60, alpha=0.55, color="tomato", label="Diabétiques", density=True)
    axes[1,1].axvline(threshold, color="black", linestyle="--", lw=2)
    axes[1,1].set_title("Distribution des Probabilités")
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "evaluation_complete.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {"auc": auc, "ap": ap, "recall_opt": recall_score(y_true, y_pred, pos_label=1, zero_division=0)}


# ==============================================================================
# BLOC 2 — ANALYSE DES BIAIS
# ==============================================================================

def _compute_robust_metrics(y_true, y_probs, y_pred, mask, total_tests=1):
    """
    Calcule les métriques avec intervalles de confiance et correction de Bonferroni.
    Le groupe cible est comparé au *reste* de la population.
    """
    n_group = int(mask.sum())
    n_pos_group = int(y_true[mask].sum())
    n_neg_group = n_group - n_pos_group
    
    if n_pos_group == 0 or n_neg_group == 0:
        return None # Impossible de calculer TPR ou FPR proprement

    yt = y_true[mask]
    yp = y_pred[mask]
    yp_prob = y_probs[mask]
    
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    
    tpr = tp / n_pos_group
    fpr = fp / n_neg_group
    
    # Erreur standard pour intervalles de confiance (95%) : Z = 1.96
    ci_tpr = 1.96 * np.sqrt((tpr * (1 - tpr)) / n_pos_group) if n_pos_group > 0 else 0
    ci_fpr = 1.96 * np.sqrt((fpr * (1 - fpr)) / n_neg_group) if n_neg_group > 0 else 0

    # Z-test : Comparer le groupe au RESTE de la population
    mask_rest = ~mask
    n_pos_rest = int(y_true[mask_rest].sum())
    tp_rest = int(((y_pred[mask_rest] == 1) & (y_true[mask_rest] == 1)).sum())
    
    if n_pos_rest > 0:
        count = np.array([tp, tp_rest])
        nobs = np.array([n_pos_group, n_pos_rest])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            if np.isnan(p_value): p_value = 1.0
    else:
        p_value = 1.0

    # Correction de Bonferroni
    p_value_adj = min(1.0, p_value * total_tests)

    try:
        auc = roc_auc_score(yt, yp_prob)
    except ValueError:
        auc = 0.0

    return {
        "n_total"      : n_group,
        "n_diabetiques": n_pos_group,
        "n_sains"      : n_neg_group,
        "pct_diab"     : round(100 * n_pos_group / n_group, 1),
        "ROC_AUC"      : round(auc, 4),
        "Recall"       : round(tpr, 4),
        "TPR"          : round(tpr, 4),
        "FPR"          : round(fpr, 4),
        "CI_TPR"       : round(ci_tpr, 4),
        "CI_FPR"       : round(ci_fpr, 4),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "pos_rate"     : round(yp.mean(), 4),
        "pvalue_adj"   : round(p_value_adj, 4),
        "Alerte_Recall": bool((p_value_adj < 0.05) and (tpr < (tp_rest/n_pos_rest if n_pos_rest>0 else 0)))
    }


def _plot_equalized_odds_robust(df_bias, var_name, global_tpr, global_fpr, results_dir):
    """
    Nouveau graphique "Crosshair" avec intervalles de confiance et zone de tolérance.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("husl", len(df_bias))
    
    # Tracer la zone de tolérance globale (+/- 5% autour du modèle global)
    ax.add_patch(plt.Rectangle(
        (global_fpr - 0.05, global_tpr - 0.05), 0.10, 0.10,
        fill=True, color="lightgray", alpha=0.4, lw=0, label="Marge d'équité acceptable (±5%)"
    ))

    ax.axhline(global_tpr, color="black", linestyle="--", lw=1.5, alpha=0.7)
    ax.axvline(global_fpr, color="black", linestyle="--", lw=1.5, alpha=0.7)

    for i, (_, row) in enumerate(df_bias.iterrows()):
        color = palette[i]
        label = str(row["Sous-groupe"]).replace("\n", " ")
        
        # Point central
        ax.scatter(row["FPR"], row["Recall"], color=color, s=120, zorder=5, edgecolor='black')
        
        # Intervalles de confiance (Barres d'erreur)
        ax.errorbar(row["FPR"], row["Recall"], 
                    xerr=row["CI_FPR"], yerr=row["CI_TPR"], 
                    fmt='none', ecolor=color, alpha=0.7, capsize=5, lw=2, zorder=4)
        
        # Annotation claire
        ax.annotate(
            f"{label}\n(n={row['n_diabetiques']})",
            xy=(row["FPR"], row["Recall"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if row["Alerte_Recall"] else "normal",
            color="darkred" if row["Alerte_Recall"] else "black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85)
        )

    ax.set_xlabel("FPR (Fausses Alarmes) ± 95% CI", fontsize=11)
    ax.set_ylabel("TPR (Recall / Sensibilité) ± 95% CI", fontsize=11)
    ax.set_title(
        f"Equalized Odds — {var_name}\n"
        "Les croix représentent l'incertitude statistique (Intervalle de confiance à 95%)", 
        fontsize=12, fontweight="bold"
    )
    
    ax.scatter([global_fpr], [global_tpr], color="black", marker="X", s=200, label="Modèle Global")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Limites dynamiques
    ax.set_xlim(max(0, df_bias["FPR"].min() - 0.1), min(1, df_bias["FPR"].max() + 0.15))
    ax.set_ylim(max(0, df_bias["Recall"].min() - 0.15), min(1, df_bias["Recall"].max() + 0.15))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_equalized_odds.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix_by_group(df_bias, var_name, results_dir):
    n_groups = len(df_bias)
    ncols = min(3, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    fig.suptitle(f"Matrices de Confusion par sous-groupe — {var_name}", fontsize=12, fontweight="bold")
    axes_flat = np.array(axes).flatten() if n_groups > 1 else [axes]

    for idx, (_, row) in enumerate(df_bias.iterrows()):
        ax = axes_flat[idx]
        cm_group = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        
        sns.heatmap(cm_group, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Prédit Sain", "Prédit Diab."], yticklabels=["Vrai Sain", "Vrai Diab."],
                    annot_kws={"size": 12, "weight": "bold"}, linewidths=0.5, linecolor="white")
        
        ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=True, color="#FFCCCC", alpha=0.6, zorder=3))
        ax.text(0.5, 1.5, f"FN = {row['FN']}", ha="center", va="center", fontsize=9, color="#8B0000", fontweight="bold", zorder=4)

        ax.set_title(f"{row['Sous-groupe']}\nRecall = {row['Recall']:.2f} | n={row['n_diabetiques']}", 
                     fontsize=10, color="red" if row["Alerte_Recall"] else "black")

    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()


def bloc2_bias_analysis(y_true, y_probs, threshold, df_test_raw, results_dir, global_metrics):
    print("\n" + "="*60)
    print("  BLOC 2 — ANALYSE DES BIAIS ")
    print("="*60)

    y_pred = (y_probs >= threshold).astype(int)
    global_recall = global_metrics["recall_opt"]
    global_auc    = global_metrics["auc"]
    
    tn_g = int(((y_pred == 0) & (y_true == 0)).sum())
    fp_g = int(((y_pred == 1) & (y_true == 0)).sum())
    global_fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0

    df_groups = reconstruct_groups_robust(df_test_raw)
    variables  = ["Age", "Sex", "Income", "Education", "GenHlth"]
    all_alerts = []

    for var in variables:
        if var not in df_groups.columns: continue
        print(f"\n  ── Variable : {var} ──")
        col = df_groups[var]
        groups = col.dropna().unique()
        
        total_tests = len(groups) # Pour Bonferroni
        rows = []
        
        for group in groups:
            mask = (col == group).values
            m = _compute_robust_metrics(y_true, y_probs, y_pred, mask, total_tests)
            if m is None: continue
            
            m["Sous-groupe"] = group
            rows.append(m)

        if not rows: continue
        df_bias = pd.DataFrame(rows)

        # Disparate Impact
        ref_pos_rate = df_bias["pos_rate"].max()
        df_bias["DI"] = (df_bias["pos_rate"] / ref_pos_rate).round(4) if ref_pos_rate > 0 else 0.0

        # Tris et Affichage
        df_bias = df_bias.sort_values("Recall", ascending=True).reset_index(drop=True)
        cols_show = ["Sous-groupe", "n_diabetiques", "Recall", "CI_TPR", "FPR", "DI", "pvalue_adj", "Alerte_Recall"]
        print(df_bias[cols_show].to_string(index=False))

        df_bias.to_csv(os.path.join(results_dir, f"biais_{var}.csv"), index=False)

        # Graphiques
        _plot_equalized_odds_robust(df_bias, var, global_recall, global_fpr, results_dir)
        _plot_confusion_matrix_by_group(df_bias, var, results_dir)

        # Alertes
        alerts = df_bias[df_bias["Alerte_Recall"]][["Sous-groupe", "Recall", "DI", "pvalue_adj"]].copy()
        if not alerts.empty:
            alerts.insert(0, "Variable", var)
            all_alerts.append(alerts)

    print("\n" + "="*60)
    print("  TABLEAU D'ALERTES (Z-test Bonferroni)")
    print("="*60)
    if all_alerts:
        df_alerts = pd.concat(all_alerts, ignore_index=True)
        df_alerts["Recall_global"] = round(global_recall, 4)
        print(df_alerts.to_string(index=False))
        df_alerts.to_csv(os.path.join(results_dir, "alertes_biais_global.csv"), index=False)
    else:
        print("  Aucun biais statistiquement significatif détecté (avec correction de Bonferroni).")


# ==============================================================================
# BLOC 3 — EMPREINTE CARBONE & TEMPS D'EXÉCUTION
# ==============================================================================

def bloc3_carbon(config, inference_emissions_kg, inference_duration_s, total_duration_s, results_dir):
    print("\n" + "="*60)
    print("  BLOC 3 — EMPREINTE CARBONE")
    print("="*60)

    power_w    = DEVICE_POWER_W[config["TRAINING_DEVICE"]]
    factor     = EMISSION_FACTORS.get(config["COUNTRY_CODE"], EMISSION_FACTORS["FR"])
    
    training_co2_g  = (power_w * config["OPTUNA_DURATION_H"] / 1000) * factor
    inference_co2_g = inference_emissions_kg * 1000
    total_co2_g     = training_co2_g + inference_co2_g

    print(f"  Empreinte inférence (CodeCarbon)    : {inference_co2_g:.4f} g")
    print(f"  Empreinte entraînement (estimation) : {training_co2_g:.2f} g")
    print(f"  TOTAL CO₂ projet                    : {total_co2_g:.2f} g CO₂eq\n")

    # [Le reste du code de visualisation de l'empreinte carbone reste identique]
    # (omis ici par pure concision pour l'affichage terminal, mais gardez 
    # votre graphique à barres horizontal pour les comparaisons).


# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

def main():
    script_start = time.perf_counter()

    print("\n" + "="*60)
    print("  ÉVALUATION FINALE")
    print("="*60)

    results_dir = setup(CONFIG)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(CONFIG, device)
    _, _, test_loader = get_dataloaders(CONFIG["TRAIN_PATH"], CONFIG["VAL_PATH"], CONFIG["TEST_PATH"], batch_size=CONFIG["BATCH_SIZE"])
    
    print("[INFO] Inférence en cours (CodeCarbon actif)...")
    tracker = EmissionsTracker(project_name="diabetes_inference", output_dir=results_dir, log_level="error", save_to_file=True)
    
    y_true, y_probs, inference_duration_s = get_predictions(model, test_loader, device, tracker=tracker)

    emissions_csv = os.path.join(results_dir, "emissions.csv")
    inference_emissions = pd.read_csv(emissions_csv)["emissions"].iloc[-1] if os.path.exists(emissions_csv) else 0.0

    # BLOC 1
    global_metrics = bloc1_evaluation(y_true, y_probs, CONFIG["OPTIMAL_THRESHOLD"], results_dir)

    # BLOC 2
    df_test_raw = pd.read_csv(CONFIG["TEST_PATH"])
    bloc2_bias_analysis(y_true, y_probs, CONFIG["OPTIMAL_THRESHOLD"], df_test_raw, results_dir, global_metrics)

    # BLOC 3
    bloc3_carbon(CONFIG, inference_emissions, inference_duration_s, time.perf_counter() - script_start, results_dir)

if __name__ == "__main__":
    main()