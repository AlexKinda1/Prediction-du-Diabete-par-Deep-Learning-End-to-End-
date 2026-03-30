# Projet-deep-_learning

Projet_deep_learning/
├── .github/workflows/
│   ├── ci.yaml                 # Tests et Linting (Black/Ruff) auto sur GitHub
│   └── cd.yaml                 # Build des images Docker et déploiement
├── configs/                    # Géré par Hydra
│   ├── model/resnet.yaml
│   ├── dataset/imagenet.yaml
│   └── config.yaml             # Point d'entrée des hyperparamètres
├── data/                       # Toujours ignoré par Git !
├── deployment/                 # Zone de production (Inférence)
│   ├── app.py                  # Point d'entrée FastAPI (démarre le serveur REST)
│   ├── model_loader.py         # Script pour télécharger le modèle depuis le Model Registry (MLflow/W&B)
│   └── schemas.py              # Classes Pydantic pour valider les requêtes/réponses de l'API
├── src/                        # Le cœur du réacteur (Recherche & Logique)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py     # Script ETL : télécharge et nettoie les données brutes (lancé par DVC)
│   │   └── datamodules.py      # Classes PyTorch Dataset et DataLoader (ou PyTorch Lightning DataModule)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py    # Vos réseaux de neurones (classes héritant de nn.Module)
│   │   └── components.py       # Blocs réutilisables (ex: couches d'Attention custom)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py            # La boucle d'entraînement principale (ou Trainer PyTorch Lightning)
│   │   ├── losses.py           # Fonctions de coût personnalisées (ex: Focal Loss)
│   │   └── optimizers.py       # Configuration des optimiseurs et schedulers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Calcul des métriques (Accuracy, F1-score)
│   │   └── predict.py          # Script pour faire des inférences en batch sur de nouvelles données
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Configuration de l'experiment tracker (MLflow, Weights & Biases)
│       └── seed.py             # Fixe les seeds (PyTorch, Numpy) pour la reproductibilité
├── tests/                      # Pytest
│   ├── test_data.py            # Vérifie que les données n'ont pas de NaNs, ont la bonne shape
│   ├── test_models.py          # Vérifie qu'un forward pass sur le modèle ne crashe pas
│   └── test_api.py             # Teste les endpoints FastAPI
├── dvc.yaml                    # Pipeline data (lie make_dataset.py, train.py, etc.)
├── Dockerfile.train            # Image lourde (CUDA, PyTorch) pour entraîner
├── Dockerfile.serve            # Image légère (ONNX Runtime, FastAPI) pour déployer
├── requirements.txt            # (Ou pyproject.toml)
└── README.md
