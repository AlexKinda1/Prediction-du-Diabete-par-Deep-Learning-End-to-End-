"""
  Ce script est organisé en 3 blocs indépendants :

  BLOC 1 — Évaluation standard sur le jeu de test
           Matrice de confusion, ROC, Précision-Rappel (épuré)

  BLOC 2 — Analyse des biais par sous-groupes (version robuste)
           Variables et méthodes améliorées :
             • Groupes reconstruits dynamiquement via pd.cut
             • Statistiques rigoureuses (Z-test)
             • Equalized Odds robuste avec Intervalles de Confiance
             • Matrices de confusion cliniques par groupe (sans ombrage)
             • Visualisation du Disparate Impact (règle des 80%)

  BLOC 3 — Empreinte carbone & temps d'exécution
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
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

# ── Chemin vers les modules du projet ────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.architectures import DiabetesMLP
from data.datamodules import get_dataloaders


# ==============================================================================
# 0. CONFIGURATION GLOBALE & CHARGEMENT DYNAMIQUE
# ==============================================================================
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# Chemin vers le dossier contenant le modèle et ses paramètres
model_dir = os.path.join(project_root, "src/training/resultats_optuna_mlflow")
params_file = os.path.join(model_dir, "optuna_params.json") 

# Fonction pour charger la configuration dynamiquement
def load_dynamic_config():
    # Valeurs par défaut au cas où le fichier JSON n'existe pas encore
    config = {
        "MODEL_PATH"        : os.path.join(model_dir, "meilleur_modele_trial_7.pth"),
        "INPUT_DIM"         : 37,
        "HIDDEN_DIMS"       : [64, 32, 16],
        "DROPOUT_RATE"      : 0.24,
        "OPTIMAL_THRESHOLD" : 0.35,
        "TRAIN_PATH" : os.path.join(project_root, "Donnee_pretraite/diabetes_train_pretraite.csv"),
        "VAL_PATH"   : os.path.join(project_root, "Donnee_pretraite/diabetes_val_pretraite.csv"),
        "TEST_PATH"  : os.path.join(project_root, "Donnee_pretraite/diabetes_test_pretraite.csv"),
        "RESULTS_DIR": "resultats_evaluation_v3",
        "BATCH_SIZE" : 256,
        "OPTUNA_DURATION_H" : 2.0,
        "TRAINING_DEVICE"   : "cpu",
        "COUNTRY_CODE"      : "FR",
    }
    
    # Écrasement par les valeurs réelles d'Optuna si le fichier existe
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            optuna_params = json.load(f)
            config["HIDDEN_DIMS"] = optuna_params.get("hidden_dims", config["HIDDEN_DIMS"])
            config["DROPOUT_RATE"] = optuna_params.get("dropout_rate", config["DROPOUT_RATE"])
            config["OPTIMAL_THRESHOLD"] = optuna_params.get("optimal_threshold", config["OPTIMAL_THRESHOLD"])
            print(f" Paramètres chargés dynamiquement depuis {params_file}")
    else:
        print(f"Fichier {params_file} introuvable. Utilisation des valeurs par défaut.")
        
    return config

CONFIG = load_dynamic_config()

EMISSION_FACTORS = {"FR": 56, "US": 386}
DEVICE_POWER_W = {"cpu": 65, "gpu": 250}

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
    return config["RESULTS_DIR"]

def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    model = DiabetesMLP(
        input_dim    = config["INPUT_DIM"],
        hidden_dims  = config["HIDDEN_DIMS"],
        dropout_rate = config["DROPOUT_RATE"],
    ).to(device)

    if not os.path.exists(config["MODEL_PATH"]):
        raise FileNotFoundError(f" Modèle introuvable : {config['MODEL_PATH']}")

    model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=device))
    model.eval()
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
    result = pd.DataFrame(index=df.index)
    result["Age"] = pd.cut(
        df["Age"], 
        bins=[-np.inf, -0.6, 0.3, np.inf], 
        labels=["Jeunes adultes\n(18-44 ans)", "Adultes\n(45-64 ans)", "Seniors\n(65 ans et +)"]
    )
    result["Sex"] = np.where(df["Sex"] < 0.5, "Femmes", "Hommes")

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

def bloc1_evaluation(y_true, y_probs, threshold, results_dir):
    print("\n" + "="*60)
    print("  BLOC 1 — ÉVALUATION SUR LE JEU DE TEST")
    print("="*60)

    y_pred    = (y_probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_probs)
    ap  = average_precision_score(y_true, y_probs)

    # Affichage épuré sur une seule ligne (1x3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Évaluation du Modèle", fontsize=15, fontweight="bold", y=1.05)

    # 1. Matrice de confusion
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Prédit Sain", "Prédit Diab."], yticklabels=["Vrai Sain", "Vrai Diab."])
    axes[0].set_title(f"Matrice de Confusion\n(Seuil optimal = {threshold:.4f})")
    
    # 2. ROC
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr_arr, tpr_arr, color="steelblue", lw=2, label=f"Courbe ROC (AUC = {auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].fill_between(fpr_arr, tpr_arr, alpha=0.08, color="steelblue")
    axes[1].set_title("Courbe ROC")
    axes[1].legend()

    # 3. Précision-Rappel
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
    axes[2].plot(rec_arr, prec_arr, color="darkorange", lw=2, label=f"Courbe P-R (AP = {ap:.4f})")
    axes[2].fill_between(rec_arr, prec_arr, y_true.mean(), alpha=0.08, color="darkorange")
    axes[2].set_title("Courbe Précision-Rappel")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "evaluation_epuree.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {"auc": auc, "ap": ap, "recall_opt": recall_score(y_true, y_pred, pos_label=1, zero_division=0)}


# ==============================================================================
# BLOC 2 — ANALYSE DES BIAIS
# ==============================================================================

def _compute_robust_metrics(y_true, y_probs, y_pred, mask, total_tests=1):
    n_group = int(mask.sum())
    n_pos_group = int(y_true[mask].sum())
    n_neg_group = n_group - n_pos_group
    
    if n_pos_group == 0 or n_neg_group == 0:
        return None

    yt = y_true[mask]
    yp = y_pred[mask]
    
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    
    tpr = tp / n_pos_group
    fpr = fp / n_neg_group
    
    ci_tpr = 1.96 * np.sqrt((tpr * (1 - tpr)) / n_pos_group) if n_pos_group > 0 else 0
    ci_fpr = 1.96 * np.sqrt((fpr * (1 - fpr)) / n_neg_group) if n_neg_group > 0 else 0

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

    p_value_adj = min(1.0, p_value * total_tests)

    return {
        "n_total": n_group, "n_diabetiques": n_pos_group,
        "Recall": round(tpr, 4), "FPR": round(fpr, 4),
        "CI_TPR": round(ci_tpr, 4), "CI_FPR": round(ci_fpr, 4),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "pos_rate": round(yp.mean(), 4),
        "pvalue_adj": round(p_value_adj, 4),
        "Alerte_Recall": bool((p_value_adj < 0.05) and (tpr < (tp_rest/n_pos_rest if n_pos_rest>0 else 0)))
    }

def _plot_disparate_impact(df_bias, var_name, results_dir):
    """
    Visualisation du Disparate Impact (Règle des 4/5)
    """
    n = len(df_bias)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.9 + 2)))

    groups = df_bias["Sous-groupe"].astype(str)
    di     = df_bias["DI"]
    colors = ["tomato" if v < 0.8 else "mediumseagreen" for v in di]

    bars = ax.barh(groups, di, color=colors, alpha=0.85, edgecolor="white", height=0.6)

    ax.axvline(1.0, color="black", linestyle="-", lw=1.5, label="Référence (DI = 1.0)")
    ax.axvline(0.8, color="tomato", linestyle="--", lw=1.5, label="Seuil d'alerte (DI = 0.80)")
    ax.axvspan(0.8, 1.2, alpha=0.06, color="green", label="Zone équitable (≥ 0.80)")

    for bar, val in zip(bars, di):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Disparate Impact (DI)", fontsize=10)
    ax.set_title(f"Disparate Impact — {var_name}\n(DI < 0.80 = Sous-détection potentielle)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_disparate_impact.png"), dpi=150, bbox_inches="tight")
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
        
        # Le patch rouge et l'annotation rouge ont été supprimés ici comme demandé.
        
        ax.set_title(f"{row['Sous-groupe']}\nRecall = {row['Recall']:.2f} | n={row['n_diabetiques']}", 
                     fontsize=10, color="red" if row["Alerte_Recall"] else "black")

    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

def _plot_equalized_odds_robust(df_bias, var_name, global_tpr, global_fpr, results_dir):
    """
    Graphique "Crosshair" avec intervalles de confiance et zone de tolérance.
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
        color = palette[i % len(palette)]
        label = str(row["Sous-groupe"]).replace("\n", " ")
        
        # Point central
        ax.scatter(row["FPR"], row["Recall"], color=color, s=120, zorder=5, edgecolor='black')
        
        # Intervalles de confiance (Barres d'erreur)
        ax.errorbar(row["FPR"], row["Recall"], 
                    xerr=row["CI_FPR"], yerr=row["CI_TPR"], 
                    fmt='none', ecolor=color, alpha=0.7, capsize=5, lw=2, zorder=4)
        
        # Annotation claire
        alerte = row.get("Alerte_Recall", False)
        ax.annotate(
            f"{label}\n(n={row['n_diabetiques']})",
            xy=(row["FPR"], row["Recall"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if alerte else "normal",
            color="darkred" if alerte else "black",
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

def bloc2_bias_analysis(y_true, y_probs, threshold, df_test_raw, results_dir, global_metrics):
    print("\n" + "="*60)
    print("  BLOC 2 — ANALYSE DES BIAIS ")
    print("="*60)

    y_pred = (y_probs >= threshold).astype(int)
    global_recall = global_metrics["recall_opt"]
    
    # Calcul du FPR global pour les graphiques
    tn_g = int(((y_pred == 0) & (y_true == 0)).sum())
    fp_g = int(((y_pred == 1) & (y_true == 0)).sum())
    global_fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0

    # 1. Reconstruction des groupes simples
    df_groups = reconstruct_groups_robust(df_test_raw)
    
    # 2. Création des variables intersectionnelles
    # On croise le Sexe avec l'Âge, le Revenu et l'Éducation
    if "Sex" in df_groups.columns:
        if "Age" in df_groups.columns:
            df_groups["Sex_x_Age"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Age"].astype(str).str.replace("\n", " ")
        if "Income" in df_groups.columns:
            df_groups["Sex_x_Income"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Income"].astype(str)
        if "Education" in df_groups.columns:
            df_groups["Sex_x_Education"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Education"].astype(str)

    # 3. Liste complète des variables à analyser (les simples, PUIS les croisées)
    variables = [
        "Age", "Sex", "Income", "Education", "GenHlth", # Analyse classique
        "Sex_x_Age", "Sex_x_Income", "Sex_x_Education"  # Analyse intersectionnelle
    ]

    all_alerts = []

    # 4. Boucle d'analyse universelle
    for var in variables:
        if var not in df_groups.columns: continue
        print(f"\n  ── Variable : {var} ──")
        col = df_groups[var]
        groups = col.dropna().unique()
        
        total_tests = len(groups)
        rows = []
        
        for group in groups:
            mask = (col == group).values
            m = _compute_robust_metrics(y_true, y_probs, y_pred, mask, total_tests)
            if m is None: continue
            
            # SÉCURITÉ : Ignorer les micro-groupes créés par l'intersectionnalité
            # (Moins de 5 diabétiques = statistiques trop bruyantes/illisibles)
            if m["n_diabetiques"] < 5:
                print(f"  [SKIP] '{group.replace(chr(10), ' ')}' ignoré (seulement {m['n_diabetiques']} diabétiques)")
                continue
                
            m["Sous-groupe"] = group
            rows.append(m)

        if not rows: continue
        df_bias = pd.DataFrame(rows)

        # Calcul du Disparate Impact
        ref_pos_rate = df_bias["pos_rate"].max()
        df_bias["DI"] = (df_bias["pos_rate"] / ref_pos_rate).round(4) if ref_pos_rate > 0 else 0.0

        # Tris et Affichage Console
        df_bias = df_bias.sort_values("Recall", ascending=True).reset_index(drop=True)
        cols_show = ["Sous-groupe", "n_diabetiques", "Recall", "CI_TPR", "FPR", "DI", "pvalue_adj", "Alerte_Recall"]
        print(df_bias[cols_show].to_string(index=False))

        # Sauvegarde CSV individuel
        df_bias.to_csv(os.path.join(results_dir, f"biais_{var}.csv"), index=False)

        # Génération des graphiques
        _plot_disparate_impact(df_bias, var, results_dir)
        _plot_equalized_odds_robust(df_bias, var, global_recall, global_fpr, results_dir)
        _plot_confusion_matrix_by_group(df_bias, var, results_dir)

        # Collecte des alertes
        alerts = df_bias[df_bias["Alerte_Recall"]][["Sous-groupe", "Recall", "DI", "pvalue_adj"]].copy()
        if not alerts.empty:
            alerts.insert(0, "Variable", var)
            all_alerts.append(alerts)

    # Bilan global des alertes
    print("\n" + "="*60)
    print("  TABLEAU RÉCAPITULATIF DES ALERTES DE BIAIS")
    print("="*60)
    if all_alerts:
        df_alerts = pd.concat(all_alerts, ignore_index=True)
        df_alerts["Recall_global"] = round(global_recall, 4)
        print(df_alerts.to_string(index=False))
        df_alerts.to_csv(os.path.join(results_dir, "alertes_biais_global.csv"), index=False)
    else:
        print("  Aucun biais statistiquement significatif détecté.")

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
    
    tracker = EmissionsTracker(project_name="diabetes_inference", output_dir=results_dir, log_level="error", save_to_file=True)
    y_true, y_probs, inference_duration_s = get_predictions(model, test_loader, device, tracker=tracker)

    global_metrics = bloc1_evaluation(y_true, y_probs, CONFIG["OPTIMAL_THRESHOLD"], results_dir)
    df_test_raw = pd.read_csv(CONFIG["TEST_PATH"])
    bloc2_bias_analysis(y_true, y_probs, CONFIG["OPTIMAL_THRESHOLD"], df_test_raw, results_dir, global_metrics)

if __name__ == "__main__":
    main()