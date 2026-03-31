import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneGroupOut

# ===================== PATHS =====================

BASE_DIR = "/data/annapan/alxtu/per_image"
RESULTS_DIR = f"{BASE_DIR}/auto_k_full"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== CONFIG =====================

RANDOM_STATE = 42

# ===================== DATA =====================

df = (
    pd.read_csv("/data/annapan/alxtu/per_image/plan_level_engineered_features.csv")
      .dropna()
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

y = df["wound_closure_terminal"]
groups = df["plantation"]

X = df.drop(columns=[
    "plantation", "p", "t", "c", "is_control", "group",
    "baseline_timepoint",
    "wound_closure_baseline",
    "wound_closure_terminal",
    "wound_closure_delta",
    "wound_closure_slope_per_hour"
])

feature_names = list(X.columns)

# ===================== MODELS =====================

def build_rf():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

def build_extra():
    return ExtraTreesRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

# ===================== CORE =====================

def run_auto_k(model_name, model_builder):

    logo = LeaveOneGroupOut()

    fold_results = []
    feature_selection_counter = {f: 0 for f in feature_names}

    print(f"\nRunning {model_name} Auto-best_k (28-fold LOGO)")

    for fold_id, (tr, te) in enumerate(
        tqdm(list(logo.split(X, y, groups)), desc=model_name, unit="fold")
    ):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # 1️⃣ RFE ranking once
        rfe = RFE(
            estimator=model_builder(),
            n_features_to_select=1,
            step=1
        )

        rfe.fit(X_tr, y_tr)

        ranking = rfe.ranking_
        ranked_idx = np.argsort(ranking)
        ranked_features = np.array(feature_names)[ranked_idx]

        # 2️⃣ Evaluate k = 1..N using MAE
        best_k = None
        best_mae = np.inf

        for k in range(1, len(ranked_features) + 1):

            selected = ranked_features[:k]

            model = model_builder()
            model.fit(X_tr[selected], y_tr)

            y_pred = model.predict(X_te[selected])

            # Since 1 sample per fold, this is absolute error
            mae = abs(y_te.values[0] - y_pred[0])

            if mae < best_mae:
                best_k = k
                best_mae = mae

        # 3️⃣ Store fold results
        fold_results.append({
            "fold": fold_id,
            "best_k": best_k,
            "best_mae": best_mae
        })

        # 4️⃣ Count selected features
        final_selected = ranked_features[:best_k]
        for f in final_selected:
            feature_selection_counter[f] += 1

    # ===================== SUMMARY =====================

    fold_df = pd.DataFrame(fold_results)

    summary = {
        "mean_best_k": fold_df["best_k"].mean(),
        "std_best_k": fold_df["best_k"].std(),
        "mean_mae": fold_df["best_mae"].mean(),
        "var_mae": fold_df["best_mae"].var()
    }

    summary_df = pd.DataFrame([summary])

    freq_df = pd.DataFrame({
        "feature": list(feature_selection_counter.keys()),
        "n_folds_selected": list(feature_selection_counter.values())
    })

    freq_df["selection_frequency"] = (
        freq_df["n_folds_selected"] / 28
    )

    # ===================== SAVE =====================

    fold_df.to_csv(f"{RESULTS_DIR}/{model_name}_fold_results.csv", index=False)
    summary_df.to_csv(f"{RESULTS_DIR}/{model_name}_summary.csv", index=False)

    freq_df.sort_values("selection_frequency", ascending=False)\
           .to_csv(f"{RESULTS_DIR}/{model_name}_selection_frequency.csv", index=False)

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print(summary_df)

# ===================== MAIN =====================

if __name__ == "__main__":

    run_auto_k("RandomForest", build_rf)
    run_auto_k("ExtraTrees", build_extra)

    print("\n✔ Auto best_k experiment completed.")
    print(f"Results saved in: {RESULTS_DIR}")
