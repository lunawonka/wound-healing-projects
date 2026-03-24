import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ===================== CONFIG =====================

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/results_auto_k"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = "/data/annapan/alxtu/features_final/data_for_training_testing_stripped.csv"

TARGET_COL = "wound_closure"
GROUP_COL = "plantation"

DROP_TARGETS = ["wound_openess"]
DROP_FEATURES = [
    "image_id","largest_region_area","plantation","initial_area","patch",
    "timepoint","p","t","c","is_control",
]

RANDOM_STATE = 42

# ===================== DATA =====================

def load_data(path):
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

def build_rf():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

# ===================== MAIN LOGO RUN =====================

def run_all_folds():

    X, y, groups, feature_names = load_data(DATA_PATH)
    logo = LeaveOneGroupOut()

    fold_results = []
    feature_selection_counter = {f: 0 for f in feature_names}

    print(f"\nTotal folds: {logo.get_n_splits(X, y, groups)}\n")

    for fold_id, (train_idx, test_idx) in enumerate(tqdm(logo.split(X, y, groups), desc="Outer folds")):

        print(f"\n--- Fold {fold_id} ---")

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # -----------------------------
        # 1) RFE once → ranking
        # -----------------------------
        rfe = RFE(
            estimator=build_rf(),
            n_features_to_select=1,
            step=1,
        )

        rfe.fit(X_tr, y_tr)

        ranking = rfe.ranking_
        ranked_idx = np.argsort(ranking)
        ranked_features = np.array(feature_names)[ranked_idx]

        # -----------------------------
        # 2) Evaluate k=1..N
        # -----------------------------
        k_results = []

        for k in range(1, len(ranked_features)+1):

            selected = ranked_features[:k]

            model = build_rf()
            model.fit(X_tr[selected], y_tr)
            y_pred = model.predict(X_te[selected])

            k_results.append({
                "k": k,
                "r2": r2_score(y_te, y_pred),
                "mae": mean_absolute_error(y_te, y_pred),
            })

        k_df = pd.DataFrame(k_results)

        best_row = k_df.loc[k_df["r2"].idxmax()]
        best_k = int(best_row["k"])
        best_features = ranked_features[:best_k]

        # Count feature selections
        for f in best_features:
            feature_selection_counter[f] += 1

        fold_results.append({
            "fold": fold_id,
            "best_k": best_k,
            "best_r2": best_row["r2"],
            "best_mae": best_row["mae"],
        })

    # =====================
    # SUMMARY
    # =====================

    fold_df = pd.DataFrame(fold_results)

    summary = {
        "mean_best_k": fold_df["best_k"].mean(),
        "std_best_k": fold_df["best_k"].std(),
        "mean_r2": fold_df["best_r2"].mean(),
        "var_r2": fold_df["best_r2"].var(),
        "mean_mae": fold_df["best_mae"].mean(),
        "var_mae": fold_df["best_mae"].var(),
    }

    summary_df = pd.DataFrame([summary])

    feature_freq_df = pd.DataFrame({
        "feature": list(feature_selection_counter.keys()),
        "n_folds_selected": list(feature_selection_counter.values()),
    })

    feature_freq_df["selection_frequency"] = (
        feature_freq_df["n_folds_selected"] / logo.get_n_splits(X, y, groups)
    )

    # =====================
    # SAVE RESULTS
    # =====================

    fold_df.to_csv(f"{RESULTS_DIR}/auto_k_per_fold_results.csv", index=False)
    summary_df.to_csv(f"{RESULTS_DIR}/auto_k_summary.csv", index=False)
    feature_freq_df.to_csv(f"{RESULTS_DIR}/auto_k_feature_selection_frequency.csv", index=False)

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print(summary_df)
    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    run_all_folds()
