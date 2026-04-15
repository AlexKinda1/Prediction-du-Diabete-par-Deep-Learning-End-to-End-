import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")


# Imports internes
from src.evaluation.general_evaluation import bloc1_evaluation
from src.evaluation.bias_analysis import bloc2_bias_analysis
from src.evaluation.carbon_tracking import bloc3_carbon

# ── Chemin projet ─────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

sys.path.insert(0, project_root)

from src.models.architectures import DiabetesMLP
from data.datamodules import get_dataloaders


# ================= CONFIG =================
def load_dynamic_config():
    model_dir = os.path.join(project_root, "src/training/resultats_optuna_mlflow")
    params_file = os.path.join(model_dir, "optuna_params.json")

    config = {
        "MODEL_PATH": os.path.join(model_dir, "meilleur_modele_trial_7.pth"),
        "INPUT_DIM": 37,
        "HIDDEN_DIMS": [64, 32, 16],
        "DROPOUT_RATE": 0.24,
        "OPTIMAL_THRESHOLD": 0.35,
        "TRAIN_PATH": os.path.join(project_root, "Donnee_pretraite/diabetes_train_pretraite.csv"),
        "VAL_PATH": os.path.join(project_root, "Donnee_pretraite/diabetes_val_pretraite.csv"),
        "TEST_PATH": os.path.join(project_root, "Donnee_pretraite/diabetes_test_pretraite.csv"),
        "RESULTS_DIR": "resultats_evaluation_v3",
        "BATCH_SIZE": 256,
    }

    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            optuna_params = json.load(f)
            config["HIDDEN_DIMS"] = optuna_params.get("hidden_dims", config["HIDDEN_DIMS"])
            config["DROPOUT_RATE"] = optuna_params.get("dropout_rate", config["DROPOUT_RATE"])
            config["OPTIMAL_THRESHOLD"] = optuna_params.get("optimal_threshold", config["OPTIMAL_THRESHOLD"])

    return config


CONFIG = load_dynamic_config()


def setup(config):
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    return config["RESULTS_DIR"]


def load_model(config, device):
    model = DiabetesMLP(
        input_dim=config["INPUT_DIM"],
        hidden_dims=config["HIDDEN_DIMS"],
        dropout_rate=config["DROPOUT_RATE"]
    ).to(device)

    model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=device))
    model.eval()
    return model


def get_predictions(model, loader, device, tracker=None):
    all_y_true, all_y_probs = [], []

    if tracker:
        tracker.start()

    t0 = time.perf_counter()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch))

            all_y_true.extend(y_batch.cpu().numpy())
            all_y_probs.extend(probs.cpu().numpy())

    duration = time.perf_counter() - t0

    if tracker:
        tracker.stop()

    return np.array(all_y_true), np.array(all_y_probs), duration


# ================= MAIN =================
def main():
    print("\n" + "=" * 60)
    print(" ÉVALUATION FINALE")
    print("=" * 60)

    results_dir = setup(CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(CONFIG, device)

    _, _, test_loader = get_dataloaders(
        CONFIG["TRAIN_PATH"],
        CONFIG["VAL_PATH"],
        CONFIG["TEST_PATH"],
        batch_size=CONFIG["BATCH_SIZE"]
    )

    tracker = EmissionsTracker(output_dir=results_dir)

    y_true, y_probs, inference_duration_s = get_predictions(model, test_loader, device, tracker)

    emissions_csv = os.path.join(results_dir, "emissions.csv")
    if os.path.exists(emissions_csv):
        df_emissions = pd.read_csv(emissions_csv)
        inference_emissions_kg = df_emissions["emissions"].iloc[-1]
    else:
        inference_emissions_kg = 0.0


    # ===== BLOC 1 =====
    global_metrics = bloc1_evaluation(
        y_true,
        y_probs,
        CONFIG["OPTIMAL_THRESHOLD"],
        results_dir
    )

    # ===== BLOC 2 =====
    df_test = pd.read_csv(CONFIG["TEST_PATH"])

    bloc2_bias_analysis(
        y_true,
        y_probs,
        CONFIG["OPTIMAL_THRESHOLD"],
        df_test,
        results_dir,
        global_metrics
    )

    bloc3_carbon(
        inference_emissions_kg, 
        inference_duration_s,
          results_dir
    )

if __name__ == "__main__":
    main()