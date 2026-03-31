import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/data/annapan/alxtu/per_patch"
RESULTS_DIR = f"{BASE_DIR}/rfe_k13_models"

raw_path = f"{RESULTS_DIR}/rfe_k13_raw.csv"
summary_path = f"{RESULTS_DIR}/rfe_k13_summary.csv"

df = pd.read_csv(raw_path)
summary = pd.read_csv(summary_path)

# ===================== R2 DISTRIBUTION =====================

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="model", y="r2")
plt.title("R² Distribution Across 28 Folds")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/r2_distribution.png")
plt.close()

# ===================== MAE DISTRIBUTION =====================

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="model", y="mae")
plt.title("MAE Distribution Across 28 Folds")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mae_distribution.png")
plt.close()

# ===================== FEATURE STABILITY =====================

plt.figure(figsize=(10,8))
sns.barplot(
    data=summary,
    x="selection_frequency",
    y="feature",
    hue="model"
)
plt.title("Feature Selection Frequency (k=13)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_stability.png")
plt.close()

print("✔ Plots saved in:", RESULTS_DIR)
