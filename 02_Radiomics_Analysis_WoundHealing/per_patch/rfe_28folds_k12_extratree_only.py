import os
import sys
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.tree import ExtraTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/results/rand_tree_res"
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)

LOG_FILE = f"{RESULTS_DIR}/rfe_28folds_k12.log"

# ===================== CONFIG =====================

DATA_PATH = "/data/annapan/alxtu/features_final/data_for_training_testing_stripped.csv"

TARGET_COL = "wound_closure"
GROUP_COL = "plantation"

DROP_TARGETS = ["wound_openess"]
DROP_FEATURES = [
    "image_id",
    "largest_region_area",
    "plantation",
    "initial_area",
    "patch",
    "timepoint",
    "p",
    "t",
    "c",
    "is_control",
]

RANDOM_STATE = 42
K_FEATURES = 12

# ===================== LOGGING =====================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("=== ExtraTree RFE 28-FOLD RUN (k=12) STARTED ===")

# ===================== DATA =====================

def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
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

# ===================== MODEL =====================

def build_extratree():
    return ExtraTreeRegressor(
        random_state=RANDOM_STATE
    )

# ===================== CORE =====================

def run_model(X, y, groups, feature_names) -> pd.DataFrame:
    ckpt_path = f"{CKPT_DIR}/ExtraTree_RFE_k12.npz"

    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        return pd.DataFrame(data["rows"], columns=data["cols"])

    logo = LeaveOneGroupOut()
    rows = []

    for fold_id, (tr, te) in enumerate(
        tqdm(list(logo.split(X, y, groups)), desc="ExtraTree_RFE", unit="fold")
    ):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # ---------- RFE ----------
        rfe = RFE(
            estimator=build_extratree(),
            n_features_to_select=K_FEATURES,
            step=1,
        )
        rfe.fit(X_tr, y_tr)
        selected = X.columns[rfe.support_]

        # ---------- FINAL MODEL ----------
        model = build_extratree()
        model.fit(X_tr[selected], y_tr)
        y_pred = model.predict(X_te[selected])

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        importances = np.zeros(len(feature_names))
        if hasattr(model, "feature_importances_"):
            importances[rfe.support_] = model.feature_importances_

        for i, f in enumerate(feature_names):
            rows.append({
                "model": "ExtraTree_RFE",
                "fold": fold_id,
                "feature": f,
                "importance": float(importances[i]),
                "r2": r2,
                "mae": mae,
            })

    df = pd.DataFrame(rows)
    np.savez(ckpt_path, rows=df.values, cols=df.columns.values)
    logging.info("Checkpoint saved")

    return df

# ===================== SUMMARY =====================

def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
    total_folds = df["fold"].nunique()

    summary = (
        df.groupby("feature")
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

    raw_df = run_model(X, y, groups, feature_names)
    raw_df.to_csv(
        f"{RESULTS_DIR}/rfe_28folds_k12_raw.csv",
        index=False
    )

    summary_df = summarize_features(raw_df)
    summary_df.to_csv(
        f"{RESULTS_DIR}/rfe_feature_metrics_summary_k12.csv",
        index=False
    )

    logging.info("=== RUN COMPLETED SUCCESSFULLY ===")
    print("✔ ExtraTree RFE 28-fold completed")
    print("✔ Results saved in results/rand_tree_res/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
