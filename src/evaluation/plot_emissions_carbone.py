import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_environmental_impact():
    # Chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    emissions_path = os.path.join(project_root, "emissions.csv")
    
    # Vérification du fichier
    if not os.path.exists(emissions_path):
        print(f"Fichier introuvable : {emissions_path}. As-tu lancé train.py avec CodeCarbon ?")
        return

    # Chargement des données
    df = pd.read_csv(emissions_path)
    
    # S'il y a plusieurs exécutions (runs), on peut les comparer. 
    # Pour l'affichage, on va s'intéresser aux composants de la dernière exécution.
    latest_run = df.iloc[-1]
    
    # Conversion de l'énergie en Watts-heures (Wh) pour une lecture plus facile (1 kWh = 1000 Wh)
    cpu_energy = latest_run['cpu_energy'] * 1000
    gpu_energy = latest_run['gpu_energy'] * 1000
    ram_energy = latest_run['ram_energy'] * 1000
    
    # Préparation du graphique
    plt.figure(figsize=(12, 5))
    
    # ---------------------------------------------------------
    # Graphique 1 : Répartition de la consommation énergétique
    # ---------------------------------------------------------
    plt.subplot(1, 2, 1)
    energies = [cpu_energy, gpu_energy, ram_energy]
    labels = ['CPU', 'GPU', 'RAM']
    colors = ['#3498db', '#e74c3c', '#f1c40f']
    
    # Si le GPU n'a pas été utilisé, on l'enlève du graphique
    if gpu_energy == 0:
        energies = [cpu_energy, ram_energy]
        labels = ['CPU', 'RAM']
        colors = ['#3498db', '#f1c40f']
        
    plt.pie(energies, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=[0.05]*len(energies))
    plt.title("Répartition de la Consommation Électrique", fontsize=14, fontweight='bold')
    
    # ---------------------------------------------------------
    # Graphique 2 : Bilan Global (Temps vs Émissions)
    # ---------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.axis('off') # On cache les axes pour afficher un joli texte de synthèse
    
    emissions_grams = latest_run['emissions'] * 1000
    duration_secs = latest_run['duration']
    country = latest_run['country_name']
    
    texte_bilan = (
        f"AUDIT ENVIRONNEMENTAL (Green AI)\n\n"
        f"Localisation : {country}\n"
        f"⏱Durée de l'entraînement : {duration_secs:.2f} secondes\n"
        f"Energie totale : {(cpu_energy + gpu_energy + ram_energy):.4f} Wh\n\n"
        f"Émissions CO2 : {emissions_grams:.4f} grammes"
    )
    
    # Pour donner un ordre de grandeur amusant et parlant
    km_voiture = emissions_grams / 120 # Moyenne: 120g CO2 / km pour une voiture essence
    texte_comparaison = f"\n\nÉquivalent : Conduire {km_voiture:.5f} km en voiture."
    
    plt.text(0.1, 0.5, texte_bilan + texte_comparaison, fontsize=14, 
             verticalalignment='center', bbox=dict(boxstyle="round,pad=1", facecolor='#e8f6f3', edgecolor='#1abc9c'))

    # Sauvegarde
    output_path = os.path.join(project_root, "impact_environnemental.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Bilan environnemental généré avec succès : {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_environmental_impact()