import os
import json
import matplotlib.pyplot as plt

def bloc3_carbon(inference_emissions_kg, inference_duration_s, results_dir):
    """
    Visualise l'empreinte carbone réelle de l'inférence mesurée par CodeCarbon.
    Génère un graphique à barres épuré et un fichier JSON.
    """
    # Conversion en grammes
    inference_co2_g = inference_emissions_kg * 1000

    # ── 1. Nouvelles références adaptées à l'échelle d'une inférence ────────
    references = {
        "Recherche web standard": 0.2,
        "Chargement d'une page web": 0.5,
        "1 min de streaming vidéo": 0.6,
        "Email (sans pièce jointe)": 4.0,
        "Inférence de notre modèle\n(Jeu de test)": inference_co2_g
    }

    sorted_refs = dict(sorted(references.items(), key=lambda x: x[1]))

    # ── 2. Création de la visualisation ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Empreinte Carbone de l'Inférence (Jeu de Test)", 
                 fontsize=15, fontweight="bold", y=0.98)

    labels = list(sorted_refs.keys())
    values = list(sorted_refs.values())

    colors = ["tomato" if "modèle" in l else "steelblue" for l in labels]
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white", height=0.6)

    ax.set_xlabel("Émissions de CO₂ équivalent (grammes)", fontsize=11)
    ax.set_title("Comparaison avec des usages numériques quotidiens", fontsize=11, color="dimgray")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # ── 3. Annotations dynamiques ────────────────────────────────────────────
    for bar, val in zip(bars, values):
        if val < 0.01:
            label_txt = f"{val * 1000:.2f} mg"
        else:
            label_txt = f"{val:.3f} g"
        ax.text(val + (max(values) * 0.02), bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", fontsize=10, fontweight="bold")

    # ── 4. Encart des KPI ────────────────────────────────────────────────────
    kpi_text = (
        f"Temps d'inférence total : {inference_duration_s:.2f} s\n"
        f"Émissions mesurées : {inference_co2_g:.4f} g CO₂eq"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='lightgray')
    ax.text(0.95, 0.05, kpi_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, color="black")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "empreinte_carbone_inference.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Sauvegarde JSON ───────────────────────────────────────────────────
    carbon_report = {
        "inference_co2_g": round(inference_co2_g, 6),
        "inference_duration_s": round(inference_duration_s, 3)
    }
    with open(os.path.join(results_dir, "empreinte_carbone.json"), "w") as f:
        json.dump(carbon_report, f, indent=2)