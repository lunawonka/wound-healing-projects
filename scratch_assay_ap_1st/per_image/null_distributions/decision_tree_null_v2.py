import sys
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor


# ================= PATHS =================
DATA_PATH = r"/data/annapan/alxtu/per_image/plan_level_engineered_features.csv"
OUTPUT_PATH = r"/data/annapan/alxtu/per_image/null_distributions/decision_tree/decision_tree_feature_pvalues_v2.csv"

# ================= CONFIG =================
TARGET_COL = "wound_closure_terminal"
GROUP_COL = "plantation"

DROP_FEATURES = [
    "plantation",
    "p",
    "t",
    "c",
    "is_control",
    "group",
    "baseline_timepoint",
    "terminal_timepoint",
    "wound_closure_baseline",
    "wound_closure_terminal",
    "wound_closure_delta",
    "wound_closure_slope_per_hour",
]

RANDOM_STATE = 42
N_SHUFFLES = 100
ALPHA = 0.05

# ================= DATA =================

def load_data(path: str):

    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    if GROUP_COL not in df.columns:
        raise ValueError(f"Missing group column '{GROUP_COL}'")

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    drop_cols = [col for col in DROP_FEATURES if col in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1) & y.notna() & groups.notna()

    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    groups = groups[valid].reset_index(drop=True)

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Plantations (folds): {groups.nunique()}")

    return X, y, groups, list(X.columns)


# ================= MODEL =================
def build_dt():
    return DecisionTreeRegressor(random_state=RANDOM_STATE)


def extract_thresholds(model):

    t = model.tree_
    thresholds = defaultdict(list)

    for node_id in range(t.node_count):
        feat = int(t.feature[node_id])
        thr = float(t.threshold[node_id])
        if feat >= 0:
            thresholds[feat].append(thr)

    return thresholds


# ================= CROSS VALIDATION =================

def cross_validated_thresholds(X, y, groups, n_features):

    logo = LeaveOneGroupOut()

    mae_scores = []
    pooled = {i: [] for i in range(n_features)}

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = build_dt()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, y_pred))

        fold_thr = extract_thresholds(model)
        for feat_idx, thr_list in fold_thr.items():
            pooled[feat_idx].extend(thr_list)

    return np.array(mae_scores), pooled


# ================= NULL DISTRIBUTION =================
def null_thresholds(X, y, groups, n_features, rng):

    seeds = rng.integers(0, 1e9, size=N_SHUFFLES)

    def single_shuffle(seed):

        local_rng = np.random.default_rng(seed)
        perm = local_rng.permutation(len(y))
        y_shuffled = pd.Series(y.values[perm])

        _, pooled = cross_validated_thresholds(
            X,
            y_shuffled,
            groups,
            n_features,
        )

        return pooled

    all_null = Parallel(n_jobs=min(20, N_SHUFFLES))(
        delayed(single_shuffle)(int(seed)) for seed in seeds
    )

    pooled_null = {i: [] for i in range(n_features)}

    for d in all_null:
        for feat_idx, thr_list in d.items():
            pooled_null[feat_idx].extend(thr_list)

    return pooled_null


# ================= MAIN =================
def main():

    rng = np.random.default_rng(RANDOM_STATE)

    X, y, groups, feature_names = load_data(DATA_PATH)

    print("\nRunning Decision Tree...")
    mae_scores, actual_thr = cross_validated_thresholds(
        X,
        y,
        groups,
        len(feature_names),
    )

    print(f"Mean MAE: {np.mean(mae_scores):.6f}")

    print("\nBuilding null distributions...")
    null_thr = null_thresholds(
        X,
        y,
        groups,
        len(feature_names),
        rng,
    )

    results = []

    for idx, feature in enumerate(feature_names):

        actual = np.array(actual_thr.get(idx, []))
        null = np.array(null_thr.get(idx, []))

        if len(actual) >= 2 and len(null) >= 2:
            _, pval = ttest_ind(actual, null, equal_var=False)
        else:
            pval = np.nan

        results.append({
            "feature": feature,
            "p_value": pval,
            "significant_0_05": pval < ALPHA if not np.isnan(pval) else False,
            "n_thresholds_actual": len(actual),
            "n_thresholds_null": len(null),
            "actual_mean_threshold": np.mean(actual) if len(actual) else np.nan,
            "null_mean_threshold": np.mean(null) if len(null) else np.nan,
            "mean_mae": np.mean(mae_scores),
        })

    res_df = pd.DataFrame(results).sort_values("p_value")
    res_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
