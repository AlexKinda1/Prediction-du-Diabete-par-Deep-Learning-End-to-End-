# Projet de Prédiction du Diabète par Deep Learning (End-to-End)

Ce projet a pour objectif de développer et de déployer un modèle de Deep Learning capable de prédire l'apparition du diabète en se basant sur des données de santé. L'ensemble du projet est conçu pour être reproductible et scalable, depuis le traitement des données jusqu'au déploiement en production.

## Architecture du Projet

Le projet est structuré de manière modulaire pour séparer les différentes préoccupations (données, entraînement, déploiement) et faciliter la maintenance et l'évolution.

```
├─── .github/             # Workflows CI/CD (GitHub Actions)
├─── configs/             # Fichiers de configuration (modèle, données, etc.)
├─── data/                # Données brutes (gérées par DVC)
├─── deployment/          # Code pour le déploiement de l'API (FastAPI/Flask)
├─── Donnee_pretraite/    # Données nettoyées et prétraitées
├─── notebooks/           # Notebooks Jupyter pour l'exploration et l'analyse
├─── src/                 # Code source du projet
│    ├─── data/           # Scripts pour la création et la gestion des datasets
│    ├─── evaluation/      # Scripts pour l'évaluation du modèle et la prédiction
│    ├─── models/          # Définition des architectures de modèles et composants
│    ├─── training/        # Scripts pour l'entraînement des modèles
│    └─── utils/           # Fonctions utilitaires (logging, etc.)
├─── tests/               # Tests unitaires et d'intégration
├─── Dockerfile.serve     # Dockerfile pour l'environnement de service (API)
├─── Dockerfile.train     # Dockerfile pour l'environnement d'entraînement
├─── dvc.yaml             # Pipeline de gestion de données avec DVC
├─── requirements.txt     # Dépendances Python
└─── README.md            # Documentation du projet
```

### Description des composants clés

*   **`src`**: Cœur du projet, il contient toute la logique applicative.
    *   **`src/data`**: Scripts pour transformer les données brutes en jeux de données prêts à l'emploi pour l'entraînement.
    *   **`src/models`**: Contient les architectures des réseaux de neurones (ex: MLP, etc.), ainsi que les composants réutilisables.
    *   **`src/training`**: Scripts pour lancer l'entraînement du modèle, incluant la gestion des optimiseurs, des fonctions de perte et des callbacks.
    *   **`src/evaluation`**: Outils pour évaluer les performances du modèle et pour effectuer des prédictions sur de nouvelles données.
*   **`configs`**: Les fichiers YAML permettent de paramétrer facilement les expériences (ex: changer les hyperparamètres du modèle, sélectionner un jeu de données) sans modifier le code.
*   **`deployment`**: Une application web (probablement FastAPI) pour exposer le modèle entraîné via une API REST.
*   **`dvc.yaml` & `data/`**: Nous utilisons DVC (Data Version Control) pour versionner les données et les modèles, assurant la reproductibilité des résultats.
*   **`.github/workflows`**: Intégration Continue (CI) pour lancer les tests automatiquement et Déploiement Continu (CD) pour déployer l'application.
*   **`Dockerfile.train` & `Dockerfile.serve`**: Des conteneurs Docker sont utilisés pour créer des environnements reproductibles pour l'entraînement et le service, évitant les problèmes de compatibilité.

## Démarrage Rapide

Cette section sera complétée avec les instructions détaillées pour l'installation et l'utilisation.

### Prérequis

*   Python 3.8+
*   DVC
*   Docker

### Installation

1.  Clonez le repository :
    ```bash
    git clone <URL_DU_PROJET>
    cd <NOM_DU_PROJET>
    ```

2.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

3.  Récupérez les données versionnées avec DVC :
    ```bash
    dvc pull
    ```

## Utilisation

Cette section décrira comment lancer les différentes étapes du projet.

### 1. Prétraitement des données

```bash
# Commande pour exécuter le pipeline DVC de traitement des données
dvc repro
```

### 2. Entraînement du modèle

```bash
# Commande pour lancer l'entraînement
python src/training/train.py --config-name=config.yaml
```

### 3. Déploiement de l'API

```bash
# Construire l'image Docker
docker build -f Dockerfile.serve -t diabetes-predictor-api .

# Lancer le conteneur
docker run -p 8000:8000 diabetes-predictor-api
```

---
*Ce README est un document vivant et sera mis à jour au fur et à mesure de l'avancement du projet.*
