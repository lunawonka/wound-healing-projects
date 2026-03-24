import sys
import os
import logging
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/results"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE = f"{RESULTS_DIR}/rfe_run.log"

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
RFE_STEP = 1          # remove 1 feature per iteration
MIN_FEATURES = 1      # eliminate down to 1 feature

ModelBuilder = Callable[[], object]

# ===================== LOGGING =====================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("=== RFE 28-FOLD RUN (NO INNER CV) STARTED ===")

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

# ===================== MODELS =====================

def build_rf():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

def build_tree():
    return DecisionTreeRegressor(random_state=RANDOM_STATE)

# ===================== CORE RUN =====================

def run_model(
    model_name: str,
    model_builder: ModelBuilder,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:

    ckpt_path = f"{CHECKPOINT_DIR}/{model_name}.npz"

    if os.path.exists(ckpt_path):
        logging.info(f"Loading checkpoint for {model_name}")
        data = np.load(ckpt_path, allow_pickle=True)
        return pd.DataFrame(data["rows"], columns=data["cols"])

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc=f"Outer LOGO [{model_name}]", unit="fold")
    ):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # ---------- RFE (NO INNER CV) ----------
        rfe = RFE(
            estimator=model_builder(),
            n_features_to_select=MIN_FEATURES,
            step=RFE_STEP,
        )

        rfe.fit(X_tr, y_tr)
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask]

        # ---------- FINAL MODEL ----------
        final_model = model_builder()
        final_model.fit(X_tr[selected_features], y_tr)
        y_pred = final_model.predict(X_te[selected_features])

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        # ---------- IMPORTANCES ----------
        importances = np.zeros(len(feature_names))
        if hasattr(final_model, "feature_importances_"):
            importances[selected_mask] = final_model.feature_importances_

        for i, fname in enumerate(feature_names):
            rows.append({
                "model": model_name,
                "fold": fold_id,
                "feature": fname,
                "importance": float(importances[i]),
                "r2": float(r2),
                "mae": float(mae),
            })

    df = pd.DataFrame(rows)

    np.savez(
        ckpt_path,
        rows=df.values,
        cols=df.columns.values,
    )

    logging.info(f"Saved checkpoint for {model_name}")
    return df

# ===================== MAIN =====================

def main():
    X, y, groups, feature_names = load_data(DATA_PATH)

    rf_df = run_model(
        "RandomForest_RFE",
        build_rf,
        X, y, groups, feature_names,
    )

    tree_df = run_model(
        "DecisionTree_RFE",
        build_tree,
        X, y, groups, feature_names,
    )

    out = pd.concat([rf_df, tree_df], ignore_index=True)
    out_path = f"{RESULTS_DIR}/rfe_28folds_results.csv"
    out.to_csv(out_path, index=False)

    logging.info("=== RUN COMPLETED SUCCESSFULLY ===")
    print(f"Saved results to {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
