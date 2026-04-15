
import numpy as np
import pandas as pd

INCOME_GROUP_MAP = {
    1: "Faibles revenus", 2: "Faibles revenus", 3: "Faibles revenus",
    4: "Revenus moyens", 5: "Revenus moyens", 6: "Revenus moyens",
    7: "Revenus élevés", 8: "Revenus élevés"
}

EDUCATION_GROUP_MAP = {
    1: "(Niv.1-4)", 2: "(Niv.1-4)", 3: "(Niv.1-4)", 4: "(Niv.1-4)",
    5: "(Niv.5-6)", 6: "(Niv.5-6)"
}

GENHLTH_LABELS = {
    1: "(1)", 2: "(2)", 3: "(3)", 4: "(4)", 5: "(5)"
}


def reconstruct_groups_robust(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    # Age
    result["Age"] = pd.cut(
        df["Age"],
        bins=[-np.inf, -0.6, 0.3, np.inf],
        labels=[
            "Jeunes adultes\n(18-44 ans)",
            "Adultes\n(45-64 ans)",
            "Seniors\n(65 ans et +)"
        ]
    )

    # Sexe
    result["Sex"] = np.where(df["Sex"] < 0.5, "Femmes", "Hommes")

    # Income
    income_cols = [c for c in df.columns if c.startswith("Income_")]
    if income_cols:
        result["Income"] = (
            df[income_cols]
            .idxmax(axis=1)
            .str.extract(r"Income_(\d+)")[0]
            .astype(int)
            .map(INCOME_GROUP_MAP)
        )

    # Education
    edu_cols = [c for c in df.columns if c.startswith("Education_")]
    if edu_cols:
        result["Education"] = (
            df[edu_cols]
            .idxmax(axis=1)
            .str.extract(r"Education_(\d+)")[0]
            .astype(int)
            .map(EDUCATION_GROUP_MAP)
        )

    # GenHlth
    gen_cols = [c for c in df.columns if c.startswith("GenHlth_")]
    if gen_cols:
        result["GenHlth"] = (
            df[gen_cols]
            .idxmax(axis=1)
            .str.extract(r"GenHlth_(\d+)")[0]
            .astype(int)
            .map(GENHLTH_LABELS)
        )

    return result