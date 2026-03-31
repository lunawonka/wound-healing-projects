import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_image"
NULL_DIR = f"{BASE_DIR}/null_distributions/decision_tree"
LOG_DIR = f"{BASE_DIR}/logs"

os.makedirs(NULL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DATA_PATH = "/data/annapan/alxtu/per_image/plan_level_engineered_features.csv"

# ===================== CONFIG =====================

TARGET_COL = "wound_closure_terminal"
GROUP_COL = "plantation"

DROP_COLS = [
    "plantation",
    "p",
    "t",
    "c",
    "is_control",
    "group",
    "baseline_timepoint",
    "wound_closure_baseline",
    "wound_closure_terminal",
    "wound_closure_delta",
    "wound_closure_slope_per_hour",
]

RANDOM_STATE = 42
N_SHUFFLES = 100

# ===================== DATA =====================

def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = (
        pd.read_csv(path)
        .dropna()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    X = df.drop(columns=DROP_COLS, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1) & y.notna() & groups.notna()
    X, y, groups = X[valid], y[valid], groups[valid]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Plantations (folds): {groups.nunique()}")

    return X, y, groups, list(X.columns)

# ===================== MODEL =====================

def build_dt():
    return DecisionTreeRegressor(random_state=RANDOM_STATE)


def extract_thresholds(model: DecisionTreeRegressor) -> Dict[int, List[float]]:
    t = model.tree_
    thresholds = defaultdict(list)

    for node_id in range(t.node_count):
        feat = int(t.feature[node_id])
        thr = float(t.threshold[node_id])
        if feat >= 0:
            thresholds[feat].append(thr)

    return thresholds

# ===================== CV THRESHOLDS =====================

def cross_validated_thresholds(
    X, y, groups
):
    logo = LeaveOneGroupOut()
    pooled = {i: [] for i in range(X.shape[1])}

    r2_scores = []
    mae_scores = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = build_dt()
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        r2_scores.append(r2_score(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))

        fold_thr = extract_thresholds(model)
        for feat_idx, thr_list in fold_thr.items():
            pooled[feat_idx].extend(thr_list)

    return (
        np.array(r2_scores),
        np.array(mae_scores),
        pooled,
    )

# ===================== NULL DISTRIBUTIONS =====================

def null_thresholds(X, y, groups, rng):
    def single_shuffle(seed):
        local_rng = np.random.default_rng(seed)
        perm = local_rng.permutation(len(y))
        y_shuffled = pd.Series(y.values[perm])

        _, _, pooled = cross_validated_thresholds(X, y_shuffled, groups)
        return pooled

    seeds = rng.integers(0, 1e9, size=N_SHUFFLES)

    all_null = Parallel(n_jobs=min(30, N_SHUFFLES))(
        delayed(single_shuffle)(int(seed)) for seed in seeds
    )

    pooled_null = {i: [] for i in range(X.shape[1])}
    for d in all_null:
        for feat_idx, thr_list in d.items():
            pooled_null[feat_idx].extend(thr_list)

    return pooled_null

# ===================== MAIN ANALYSIS =====================

def main():

    rng = np.random.default_rng(RANDOM_STATE)

    X, y, groups, feature_names = load_data(DATA_PATH)

    print("\nRunning actual model...")
    r2_scores, mae_scores, actual_thr = cross_validated_thresholds(X, y, groups)

    print(f"Mean R²: {np.mean(r2_scores):.4f}")
    print(f"Mean MAE: {np.mean(mae_scores):.4f}")

    print("\nBuilding null distributions...")
    null_thr = null_thresholds(X, y, groups, rng)

    results = []

    for idx, feature in enumerate(feature_names):
        actual = np.array(actual_thr.get(idx, []))
        null = np.array(null_thr.get(idx, []))

        if len(actual) >= 2 and len(null) >= 2:
            _, pval = ttest_ind(actual, null, equal_var=False)
            pval = float(pval)
        else:
            pval = np.nan

        results.append({
            "feature": feature,
            "p_value": pval,
            "n_thresholds_actual": len(actual),
            "n_thresholds_null": len(null),
            "actual_mean": np.mean(actual) if len(actual) else np.nan,
            "null_mean": np.mean(null) if len(null) else np.nan,
            "mean_r2": float(np.mean(r2_scores)),
            "mean_mae": float(np.mean(mae_scores)),
        })

    res_df = pd.DataFrame(results).sort_values("p_value")

    out_path = f"{NULL_DIR}/decision_tree_feature_pvalues.csv"
    res_df.to_csv(out_path, index=False)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
