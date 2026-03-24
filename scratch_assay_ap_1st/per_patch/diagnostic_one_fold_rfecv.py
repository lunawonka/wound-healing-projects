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
DIAG_DIR = f"{BASE_DIR}/diagnostic_res"

os.makedirs(DIAG_DIR, exist_ok=True)

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

# ===================== DIAGNOSTIC =====================

def run_diagnostic():
    X, y, groups, feature_names = load_data(DATA_PATH)

    logo = LeaveOneGroupOut()
    train_idx, test_idx = list(logo.split(X, y, groups))[0]

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train samples: {X_tr.shape[0]}")
    print(f"Test samples : {X_te.shape[0]}")
    print(f"Features     : {X_tr.shape[1]}")
    print("-" * 60)

    # -------------------------------------------------
    # 1) RFE ONCE → FEATURE RANKING
    # -------------------------------------------------

    print("Running RFE once to get feature ranking...")

    rfe = RFE(
        estimator=build_rf(),
        n_features_to_select=1,
        step=1,
    )

    start = time.time()
    rfe.fit(X_tr, y_tr)
    print(f"RFE ranking completed in {(time.time()-start)/60:.2f} minutes")

    ranking = rfe.ranking_
    ranked_idx = np.argsort(ranking)
    ranked_features = np.array(feature_names)[ranked_idx]

    # -------------------------------------------------
    # 2) EVALUATE k = 1..N (DATA-DRIVEN)
    # -------------------------------------------------

    results = []

    print("\nEvaluating performance for k = 1..N features")

    for k in tqdm(range(1, len(ranked_features) + 1), desc="Evaluating k"):
        selected = ranked_features[:k]

        model = build_rf()
        model.fit(X_tr[selected], y_tr)
        y_pred = model.predict(X_te[selected])

        results.append({
            "k": k,
            "r2": r2_score(y_te, y_pred),
            "mae": mean_absolute_error(y_te, y_pred),
        })

    res_df = pd.DataFrame(results)

    # -------------------------------------------------
    # 3) MODEL CHOOSES BEST k
    # -------------------------------------------------

    best_row = res_df.loc[res_df["r2"].idxmax()]
    best_k = int(best_row["k"])
    best_features = ranked_features[:best_k]

    print("\n=== MODEL-SELECTED RESULT ===")
    print(f"Best k (by R²): {best_k}")
    print(f"R²  : {best_row['r2']:.4f}")
    print(f"MAE : {best_row['mae']:.4f}")

    print("\nSelected features:")
    for f in best_features:
        print("  -", f)

    # Optional: save diagnostic curve
    out_path = f"{DIAG_DIR}/diagnostic_rfe_k_curve.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved diagnostic results to {out_path}")

if __name__ == "__main__":
    run_diagnostic()
