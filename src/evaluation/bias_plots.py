
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _plot_disparate_impact(df_bias, var_name, results_dir):
    n = len(df_bias)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.9 + 2)))

    groups = df_bias["Sous-groupe"].astype(str)
    di = df_bias["DI"]

    colors = ["tomato" if v < 0.8 else "mediumseagreen" for v in di]

    bars = ax.barh(groups, di, color=colors, alpha=0.85, edgecolor="white", height=0.6)

    ax.axvline(1.0, color="black", linestyle="-", lw=1.5, label="Référence (DI = 1.0)")
    ax.axvline(0.8, color="tomato", linestyle="--", lw=1.5, label="Seuil d'alerte (DI = 0.80)")
    ax.axvspan(0.8, 1.2, alpha=0.06, color="green", label="Zone équitable (≥ 0.80)")

    for bar, val in zip(bars, di):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Disparate Impact (DI)", fontsize=10)
    ax.set_title(f"Disparate Impact — {var_name}", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_disparate_impact.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix_by_group(df_bias, var_name, results_dir):
    n_groups = len(df_bias)
    ncols = min(3, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    fig.suptitle(f"Matrices de Confusion — {var_name}", fontsize=12)

    axes_flat = np.array(axes).flatten() if n_groups > 1 else [axes]

    for idx, (_, row) in enumerate(df_bias.iterrows()):
        ax = axes_flat[idx]
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Prédit Sain", "Prédit Diab."],
                    yticklabels=["Vrai Sain", "Vrai Diab."],
                    cbar=False)

        ax.set_title(f"{row['Sous-groupe']} (Recall={row['Recall']:.2f})")

    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_confusion_matrices.png"), dpi=150)
    plt.close()


def _plot_equalized_odds_robust(df_bias, var_name, global_tpr, global_fpr, results_dir):
    """
    Génère un graphique Equalized Odds esthétique et annoté, avec intervalles de confiance.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Palette de couleurs pour différencier les groupes
    palette = sns.color_palette("husl", len(df_bias))

    # 1. Zone d'équité (Carré gris clair au centre)
    ax.add_patch(plt.Rectangle(
        (global_fpr - 0.05, global_tpr - 0.05), 
        0.10, 0.10,
        fill=True, color="lightgray", alpha=0.4, lw=0, 
        label="Marge d'équité acceptable (±5%)"
    ))

    # 2. Lignes pointillées globales
    ax.axhline(global_tpr, color="black", linestyle="--", lw=1.5, alpha=0.7)
    ax.axvline(global_fpr, color="black", linestyle="--", lw=1.5, alpha=0.7)

    # 3. Tracé des points, barres d'erreur et annotations
    for i, (_, row) in enumerate(df_bias.iterrows()):
        color = palette[i % len(palette)]
        
        # Formatage du texte (suppression des sauts de ligne pour un affichage propre)
        label_text = str(row["Sous-groupe"]).replace("\n", " ")
        n_diab = row.get("n_diabetiques", "N/A")
        
        # Point central (avec bordure noire pour bien ressortir)
        ax.scatter(row["FPR"], row["Recall"], color=color, s=120, 
                   zorder=5, edgecolors='black')

        # Barres d'erreur (moustaches)
        ax.errorbar(row["FPR"], row["Recall"], 
                    xerr=row.get("CI_FPR", 0), yerr=row.get("CI_TPR", 0), 
                    fmt='none', ecolor=color, alpha=0.7, capsize=5, lw=2, zorder=4)

        # Logique d'alerte : Rouge et gras si le groupe est discriminé
        alerte = row.get("Alerte_Recall", False)
        text_color = "darkred" if alerte else "black"
        font_weight = "bold" if alerte else "normal"

        # Annotation (La bulle de texte avec fond blanc et bordure colorée)
        ax.annotate(
            f"{label_text}\n(n={n_diab})",
            xy=(row["FPR"], row["Recall"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight=font_weight,
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
            zorder=6
        )

    # 4. Point de référence global (La grosse croix noire)
    ax.scatter([global_fpr], [global_tpr], color="black", marker="X", s=200, 
               label="Modèle Global", zorder=7)

    # 5. Esthétique générale (Titres, Axes, Grille, Légende)
    ax.set_xlabel("FPR (Fausses Alarmes) ± 95% CI", fontsize=11)
    ax.set_ylabel("TPR (Recall / Sensibilité) ± 95% CI", fontsize=11)
    ax.set_title(
        f"Equalized Odds — {var_name}\n"
        "Les croix représentent l'incertitude statistique (Intervalle de confiance à 95%)", 
        fontsize=12, fontweight="bold"
    )
    
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6) # Grille en petits pointillés

    # Ajustement dynamique des limites pour ne pas couper les boîtes de texte
    fpr_min, fpr_max = df_bias["FPR"].min(), df_bias["FPR"].max()
    tpr_min, tpr_max = df_bias["Recall"].min(), df_bias["Recall"].max()
    
    ax.set_xlim(max(0, fpr_min - 0.15), min(1, fpr_max + 0.20))
    ax.set_ylim(max(0, tpr_min - 0.15), min(1, tpr_max + 0.15))

    # Sauvegarde
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"biais_{var_name}_equalized_odds.png"), dpi=150, bbox_inches="tight")
    plt.close()