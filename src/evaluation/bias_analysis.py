
import os
import numpy as np
import pandas as pd
import warnings
from statsmodels.stats.proportion import proportions_ztest

from .bias_plots import (
    _plot_disparate_impact,
    _plot_equalized_odds_robust,
    _plot_confusion_matrix_by_group
)

from .data_utils import reconstruct_groups_robust

warnings.filterwarnings("ignore")


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

    ci_tpr = 1.96 * np.sqrt((tpr * (1 - tpr)) / n_pos_group)
    ci_fpr = 1.96 * np.sqrt((fpr * (1 - fpr)) / n_neg_group)

    mask_rest = ~mask
    n_pos_rest = int(y_true[mask_rest].sum())
    tp_rest = int(((y_pred[mask_rest] == 1) & (y_true[mask_rest] == 1)).sum())

    if n_pos_rest > 0:
        count = np.array([tp, tp_rest])
        nobs = np.array([n_pos_group, n_pos_rest])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value = proportions_ztest(count, nobs)
            if np.isnan(p_value):
                p_value = 1.0
    else:
        p_value = 1.0

    p_value_adj = min(1.0, p_value * total_tests)

    return {
        "n_total": n_group,
        "n_diabetiques": n_pos_group,
        "Recall": round(tpr, 4),
        "FPR": round(fpr, 4),
        "CI_TPR": round(ci_tpr, 4),
        "CI_FPR": round(ci_fpr, 4),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "pos_rate": round(yp.mean(), 4),
        "pvalue_adj": round(p_value_adj, 4),
        "Alerte_Recall": bool(
            (p_value_adj < 0.05) and
            (tpr < (tp_rest / n_pos_rest if n_pos_rest > 0 else 0))
        )
    }


def bloc2_bias_analysis(y_true, y_probs, threshold, df_test_raw, results_dir, global_metrics):
    print("\n" + "=" * 60)
    print("  BLOC 2 — ANALYSE DES BIAIS ")
    print("=" * 60)

    y_pred = (y_probs >= threshold).astype(int)
    global_recall = global_metrics["recall_opt"]

    tn_g = int(((y_pred == 0) & (y_true == 0)).sum())
    fp_g = int(((y_pred == 1) & (y_true == 0)).sum())
    global_fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0

    df_groups = reconstruct_groups_robust(df_test_raw)

    # Variables complètes (fidèle)
    if "Sex" in df_groups.columns:
        if "Age" in df_groups.columns:
            df_groups["Sex_x_Age"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Age"].astype(str).str.replace("\n", " ")
        if "Income" in df_groups.columns:
            df_groups["Sex_x_Income"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Income"].astype(str)
        if "Education" in df_groups.columns:
            df_groups["Sex_x_Education"] = df_groups["Sex"].astype(str) + "\n+ " + df_groups["Education"].astype(str)

    variables = [
        "Age", "Sex", "Income", "Education", "GenHlth",
        "Sex_x_Age", "Sex_x_Income", "Sex_x_Education"
    ]

    all_alerts = []

    for var in variables:
        if var not in df_groups.columns:
            continue

        print(f"\n  ── Variable : {var} ──")

        col = df_groups[var]
        groups = col.dropna().unique()

        rows = []

        for group in groups:
            mask = (col == group).values
            m = _compute_robust_metrics(y_true, y_probs, y_pred, mask, len(groups))

            if m is None:
                continue

            if m["n_diabetiques"] < 5:
                continue

            m["Sous-groupe"] = group
            rows.append(m)

        if not rows:
            continue

        df_bias = pd.DataFrame(rows)

        ref_pos_rate = df_bias["pos_rate"].max()
        df_bias["DI"] = (df_bias["pos_rate"] / ref_pos_rate).round(4)

        df_bias.to_csv(os.path.join(results_dir, f"biais_{var}.csv"), index=False)

        _plot_disparate_impact(df_bias, var, results_dir)
        _plot_equalized_odds_robust(df_bias, var, global_recall, global_fpr, results_dir)
        _plot_confusion_matrix_by_group(df_bias, var, results_dir)

        alerts = df_bias[df_bias["Alerte_Recall"]][["Sous-groupe", "Recall", "DI", "pvalue_adj"]].copy()

        if not alerts.empty:
            alerts.insert(0, "Variable", var)
            all_alerts.append(alerts)

    print("\n" + "=" * 60)
    print("  TABLEAU RÉCAPITULATIF DES ALERTES DE BIAIS")
    print("=" * 60)

    if all_alerts:
        df_alerts = pd.concat(all_alerts, ignore_index=True)
        df_alerts["Recall_global"] = round(global_recall, 4)

        print(df_alerts.to_string(index=False))
        df_alerts.to_csv(os.path.join(results_dir, "alertes_biais_global.csv"), index=False)
    else:
        print("  Aucun biais statistiquement significatif détecté.")