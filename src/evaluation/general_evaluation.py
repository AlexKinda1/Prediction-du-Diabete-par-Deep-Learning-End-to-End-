import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score
)

def bloc1_evaluation(y_true, y_probs, threshold, results_dir):
    print("\n" + "="*60)
    print("ÉVALUATION SUR LES DONNEES DE TEST")
    print("="*60)

    y_pred = (y_probs >= threshold).astype(int)

    auc  = roc_auc_score(y_true, y_probs)
    ap   = average_precision_score(y_true, y_probs)
    acc  = (y_pred == y_true).mean()
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"\n  [MÉTRIQUES GLOBALES - Seuil : {threshold:.4f}]")
    print(f"  • ROC AUC   : {auc:.4f}")
    print(f"  • Avg Prec  : {ap:.4f}")
    print(f"  • Accuracy  : {acc:.4f}")
    print(f"  • Recall    : {rec:.4f}")
    print(f"  • Precision : {prec:.4f}")
    print(f"  • F1-Score  : {f1:.4f}\n")

    report_text = classification_report(
        y_true, y_pred,
        target_names=["Sain (0)", "Diabétique (1)"],
        zero_division=0
    )

    with open(os.path.join(results_dir, "rapport_evaluation.txt"), "w", encoding="utf-8") as f:
        f.write(f"MÉTRIQUES GLOBALES (Seuil = {threshold:.4f})\n")
        f.write("-" * 40 + "\n")
        f.write(f"ROC AUC           : {auc:.4f}\n")
        f.write(f"Average Precision : {ap:.4f}\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"F1-Score          : {f1:.4f}\n\n")
        f.write(report_text)

    # Initialisation de la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Évaluation Globale du Modèle", fontsize=16, fontweight="bold", y=1.05)

    # 1. Matrice de Confusion annotée
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Prédit Sain", "Prédit Diab."],
                yticklabels=["Vrai Sain", "Vrai Diab."])
    axes[0].set_title(f"Matrice de Confusion\n(Seuil optimal = {threshold:.4f})", fontsize=12)
    axes[0].set_xlabel("Prédictions du Modèle", fontsize=11)
    axes[0].set_ylabel("Réalité (Vérité Terrain)", fontsize=11)

    # 2. Courbe ROC annotée
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr_arr, tpr_arr, color="steelblue", lw=2, label=f"Courbe ROC (AUC = {auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    axes[1].fill_between(fpr_arr, tpr_arr, alpha=0.08, color="steelblue")
    axes[1].set_title("Courbe ROC\n(Capacité de discrimination)", fontsize=12)
    axes[1].set_xlabel("Taux de Faux Positifs (FPR)", fontsize=11)
    axes[1].set_ylabel("Taux de Vrais Positifs (TPR / Recall)", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Courbe Précision-Rappel annotée
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
    axes[2].plot(rec_arr, prec_arr, color="darkorange", lw=2, label=f"Courbe P-R (AP = {ap:.4f})")
    axes[2].fill_between(rec_arr, prec_arr, y_true.mean(), alpha=0.08, color="darkorange")
    axes[2].set_title("Courbe Précision-Rappel\n(Performance sur classes déséquilibrées)", fontsize=12)
    axes[2].set_xlabel("Rappel (Recall / Sensibilité)", fontsize=11)
    axes[2].set_ylabel("Précision", fontsize=11)
    axes[2].legend(loc="lower left", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "evaluation_epuree.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {"auc": auc, "ap": ap, "recall_opt": rec}