import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/results"
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)

LOG_FILE = f"{RESULTS_DIR}/rfe_28folds.log"

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
K_FEATURES = 12

ModelBuilder = Callable[[], object]

# ===================== LOGGING =====================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.info("=== RFE 28-FOLD FINAL RUN (k=12) STARTED ===")

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

# ===================== CORE =====================

def run_model(
    model_name: str,
    model_builder: ModelBuilder,
    X, y, groups, feature_names
) -> pd.DataFrame:

    ckpt_path = f"{CKPT_DIR}/{model_name}_k12.npz"
    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        return pd.DataFrame(data["rows"], columns=data["cols"])

    logo = LeaveOneGroupOut()
    rows = []

    for fold_id, (tr, te) in enumerate(
        tqdm(list(logo.split(X, y, groups)), desc=f"{model_name}", unit="fold")
    ):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # RFE → ranking
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
        if hasattr(model, "feature_importances_"):
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

    df = pd.DataFrame(rows)
    np.savez(ckpt_path, rows=df.values, cols=df.columns.values)
    return df

# ===================== SUMMARY =====================

def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
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
    tree_df = run_model("DecisionTree", build_tree, X, y, groups, feature_names)

    raw_df = pd.concat([rf_df, tree_df], ignore_index=True)
    raw_df.to_csv(f"{RESULTS_DIR}/rfe_28folds_raw.csv", index=False)

    summary_df = summarize_features(raw_df)
    summary_df.to_csv(
        f"{RESULTS_DIR}/rfe_feature_metrics_summary.csv",
        index=False
    )

    logging.info("=== RUN COMPLETED SUCCESSFULLY ===")
    print("✔ 28-fold RFE completed")
    print("✔ Results saved in results/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
