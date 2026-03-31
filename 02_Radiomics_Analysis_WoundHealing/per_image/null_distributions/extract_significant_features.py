import os
import pandas as pd

BASE_DIR = "/data/annapan/alxtu/per_image"
NULL_DIR = f"{BASE_DIR}/null_distributions/decision_tree"
CONS_DIR = f"{BASE_DIR}/consensus"

os.makedirs(CONS_DIR, exist_ok=True)

PVAL_FILE = f"{NULL_DIR}/decision_tree_feature_pvalues.csv"
P_THRESHOLD = 0.05


def main():

    df = pd.read_csv(PVAL_FILE)

    sig = df[df["p_value"] < P_THRESHOLD].copy()
    sig = sig.sort_values("p_value")

    K = len(sig)

    print("\n==============================")
    print(" SIGNIFICANT FEATURES (p < 0.05)")
    print("==============================\n")

    print(f"Total significant features (K): {K}\n")

    print(sig[["feature", "p_value"]])

    out_path = f"{CONS_DIR}/significant_features_p_lt_0_05.csv"
    sig.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
