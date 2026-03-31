import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("/data/annapan/alxtu/eng_feat_v3/res_eng/rf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = (
    pd.read_excel("/data/annapan/alxtu/eng_feat_v3/plan_level_engineered_features.xlsx")
      .dropna()
      .sample(frac=1, random_state=42)  
      .reset_index(drop=True)
)
y = df["wound_closure_terminal"]
X = df.drop(columns=["plantation", "p", "t", "c", "is_control" , "group", "baseline_timepoint", "wound_closure_baseline", "wound_closure_terminal", "wound_closure_delta", "wound_closure_slope_per_hour"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Top 20 Features
top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(x="importance", y="feature", data=feature_importances.head(top_n))
plt.title(f"Top {top_n} important feature for wound_closure with Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUT_DIR / "RF_plot1.png")
plt.show()

# save results
feature_importances.to_csv(OUT_DIR / "RF_eng_results_all.csv", index=False)

# After saving feature_importances.to_csv("RF_eng_results_all.csv", index=False)
rf = pd.read_csv(OUT_DIR / "RF_eng_results_all.csv").sort_values(by="importance", ascending=False)
rf["cumulative_importance"] = rf["importance"].cumsum()

# Define threshold (e.g., top features explaining 80% of total importance)
threshold = 0.8
core_features = rf[rf["cumulative_importance"] <= threshold]

print(f"\n✅ {len(core_features)} features explain {threshold*100:.0f}% of the total model importance.")
print(core_features)

plt.figure(figsize=(8,5))
plt.plot(range(len(rf)), rf["cumulative_importance"], marker="o")
plt.axhline(y=threshold, color='r', linestyle='--', label=f"{threshold*100:.0f}% threshold")
plt.xlabel("Number of features (sorted by importance)")
plt.ylabel("Cumulative Importance")
plt.title("Cumulative Contribution of Feature Importances with RF")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "RF_plot2.png")
plt.show()

# Save the core features (those explaining 80% of importance)
core_features.to_csv(OUT_DIR/ "RF_results_eng.csv", index=False)
print(f"\n💾 The {len(core_features)} most important features have been saved to 'RF_results_eng.csv'")
