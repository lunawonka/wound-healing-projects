import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
df = (
    pd.read_csv("/data/annapan/alxtu/features_final/data_for_training_testing_stripped_reduced.csv")
      .dropna()
      .sample(frac=1, random_state=42)  
      .reset_index(drop=True)
)

y = df["wound_closure"]
X = df.drop(columns=["wound_closure", "wound_openess"])

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
plt.title(f"Top {top_n} Σημαντικότερα Χαρακτηριστικά για wound_closure")
plt.xlabel("Σημασία")
plt.ylabel("Χαρακτηριστικό")
plt.tight_layout()
plt.show()

# save results
feature_importances.to_csv("RF_results.csv", index=False)

# After saving feature_importances.to_csv("RF_results.csv", index=False)
rf = pd.read_csv("RF_results.csv").sort_values(by="importance", ascending=False)
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
plt.title("Cumulative Contribution of Feature Importances")
plt.legend()
plt.tight_layout()
plt.savefig("RF_newplot.png")
plt.show()


# Save the core features (those explaining 80% of importance)
core_features.to_csv("RF_results_new.csv", index=False)
print(f"\n💾 The {len(core_features)} most important features have been saved to 'RF_results_new.csv'")
