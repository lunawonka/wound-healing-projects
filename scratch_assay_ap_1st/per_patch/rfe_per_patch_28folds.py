import sys
import os
import logging
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/results"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
N_SHUFFLES = 100
PARALLEL_JOBS = 10

LOG_FILE = f"{RESULTS_DIR}/rfe_run.log"

ModelBuilder = Callable[[], object]

# ===================== LOGGING =====================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("=== RFE + NULL IMPORTANCE RUN STARTED ===")

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

# ===================== CV + RFE =====================

def cv_rfe_importances(model_builder, X, y, groups):
    logo = LeaveOneGroupOut()
    r2_scores, mae_scores, importances = [], [], []

    splits = list(logo.split(X, y, groups))
    for train_idx, test_idx in tqdm(
        splits, desc="Outer LOGO folds", leave=False
    ):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        g_tr = groups.iloc[train_idx]

        rfe = RFECV(
            estimator=model_builder(),
            step=1,
            cv=LeaveOneGroupOut(),
            scoring="r2",
            n_jobs=1,
        )

        rfe.fit(X_tr, y_tr, groups=g_tr)
        selected = X.columns[rfe.support_]

        final_model = model_builder()
        final_model.fit(X_tr[selected], y_tr)

        y_pred = final_model.predict(X_te[selected])

        r2_scores.append(r2_score(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))

        full_imp = np.zeros(X.shape[1])
        full_imp[rfe.support_] = final_model.feature_importances_
        importances.append(full_imp)

    return np.array(r2_scores), np.array(mae_scores), np.vstack(importances)

# ===================== NULL IMPORTANCES =====================

def null_importances(model_name, model_builder, X, y, groups, rng):
    checkpoint_path = f"{CHECKPOINT_DIR}/{model_name}_null.npz"

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading null checkpoint for {model_name}")
        return np.load(checkpoint_path)["imps"]

    seeds = rng.integers(0, 2**32 - 1, size=N_SHUFFLES)

    def _single(seed):
        y_shuf = pd.Series(
            np.random.default_rng(seed).permutation(y.values),
            index=y.index,
        )
        _, _, imps = cv_rfe_importances(model_builder, X, y_shuf, groups)
        return imps

    all_imps = Parallel(
        n_jobs=min(PARALLEL_JOBS, N_SHUFFLES),
        prefer="processes",
    )(
        delayed(_single)(int(s))
        for s in tqdm(seeds, desc=f"Null shuffles [{model_name}]")
    )

    all_imps = np.vstack(all_imps)
    np.savez(checkpoint_path, imps=all_imps)
    logging.info(f"Saved null checkpoint for {model_name}")

    return all_imps

# ===================== SUMMARY =====================

def summarize_model(model_name, model_builder, X, y, groups, feature_names, rng):
    logging.info(f"Running model {model_name}")

    actual_path = f"{CHECKPOINT_DIR}/{model_name}_actual.npz"

    if os.path.exists(actual_path):
        logging.info(f"Loading actual checkpoint for {model_name}")
        data = np.load(actual_path)
        r2, mae, actual_imps = data["r2"], data["mae"], data["imps"]
    else:
        r2, mae, actual_imps = cv_rfe_importances(model_builder, X, y, groups)
        np.savez(actual_path, r2=r2, mae=mae, imps=actual_imps)
        logging.info(f"Saved actual checkpoint for {model_name}")

    null_imps = null_importances(model_name, model_builder, X, y, groups, rng)

    rows = []
    for i, f in enumerate(feature_names):
        _, p = ttest_ind(
            actual_imps[:, i],
            null_imps[:, i],
            equal_var=False,
        )

        rows.append({
            "model": model_name,
            "feature": f,
            "p_value": float(p),
            "actual_mean_importance": float(actual_imps[:, i].mean()),
            "null_mean_importance": float(null_imps[:, i].mean()),
            "r2_mean": float(r2.mean()),
            "r2_variance": float(np.var(r2, ddof=1)),
            "mae_mean": float(mae.mean()),
            "mae_variance": float(np.var(mae, ddof=1)),
        })

    return pd.DataFrame(rows)

# ===================== MAIN =====================

def main():
    X, y, groups, feature_names = load_data(DATA_PATH)
    rng = np.random.default_rng(RANDOM_STATE)

    rf_df = summarize_model(
        "RandomForest_RFE", build_rf, X, y, groups, feature_names, rng
    )
    tree_df = summarize_model(
        "DecisionTree_RFE", build_tree, X, y, groups, feature_names, rng
    )

    out = pd.concat([rf_df, tree_df], ignore_index=True)
    out_path = f"{RESULTS_DIR}/rfe_feature_significance.csv"
    out.to_csv(out_path, index=False)

    logging.info("=== RUN COMPLETED SUCCESSFULLY ===")
    print(f"Saved {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
