import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Tuple

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/rfe_k13_models"
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ===================== CONFIG =====================

DATA_PATH = "/data/annapan/alxtu/features_final/data_for_training_testing_stripped.csv"

TARGET_COL = "wound_closure"
GROUP_COL = "plantation"

DROP_TARGETS = ["wound_openess"]
DROP_FEATURES = [
    "image_id","largest_region_area","plantation","initial_area","patch",
    "timepoint","p","t","c","is_control",
]

RANDOM_STATE = 42
K_FEATURES = 13

ModelBuilder = Callable[[], object]

# ===================== DATA =====================

def load_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_TARGETS if c in df.columns])

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    X = df.drop(columns=[TARGET_COL] + DROP_FEATURES, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1) & y.notna() & groups.notna()
    X, y, groups = X[valid], y[valid], groups[valid]

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        groups.reset_index(drop=True),
        list(X.columns),
    )

# ===================== MODELS =====================

def build_rf():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

def build_et():
    return ExtraTreesRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

# ===================== CORE =====================

def run_model(model_name, model_builder, X, y, groups, feature_names):

    logo = LeaveOneGroupOut()
    rows = []

    for fold_id, (tr, te) in enumerate(
        tqdm(list(logo.split(X, y, groups)), desc=model_name, unit="fold")
    ):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        rfe = RFE(
            estimator=model_builder(),
            n_features_to_select=K_FEATURES,
            step=1,
        )
        rfe.fit(X_tr, y_tr)

        selected = X.columns[rfe.support_]

        model = model_builder()
        model.fit(X_tr[selected], y_tr)
        y_pred = model.predict(X_te[selected])

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        importances = np.zeros(len(feature_names))
        importances[rfe.support_] = model.feature_importances_

        for i, f in enumerate(feature_names):
            rows.append({
                "model": model_name,
                "fold": fold_id,
                "feature": f,
                "importance": float(importances[i]),
                "r2": r2,
                "mae": mae,
            })

    return pd.DataFrame(rows)

# ===================== SUMMARY =====================

def summarize(df):

    total_folds = df["fold"].nunique()

    summary = (
        df.groupby(["model", "feature"])
        .agg(
            mean_importance=("importance", "mean"),
            var_importance=("importance", "var"),
            mean_r2=("r2", "mean"),
            var_r2=("r2", "var"),
            mean_mae=("mae", "mean"),
            var_mae=("mae", "var"),
            n_folds_selected=("importance", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    summary["selection_frequency"] = (
        summary["n_folds_selected"] / total_folds
    )

    return summary

# ===================== MAIN =====================

def main():
    X, y, groups, feature_names = load_data(DATA_PATH)

    rf_df = run_model("RandomForest", build_rf, X, y, groups, feature_names)
    et_df = run_model("ExtraTree", build_et, X, y, groups, feature_names)

    raw_df = pd.concat([rf_df, et_df], ignore_index=True)
    raw_df.to_csv(f"{RESULTS_DIR}/rfe_k13_raw.csv", index=False)

    summary_df = summarize(raw_df)
    summary_df.to_csv(f"{RESULTS_DIR}/rfe_k13_summary.csv", index=False)

    print("✔ RFE k=13 completed")
    print(f"✔ Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    sys.exit(main())
