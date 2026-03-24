import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

OUT_DIR = Path("/data/annapan/alxtu/eng_feat_v3/res_eng/rfe_rf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = (
    pd.read_excel("/data/annapan/alxtu/eng_feat_v3/plan_level_engineered_features.xlsx")
      .dropna()
      .sample(frac=1, random_state=42)  
      .reset_index(drop=True)
)

X = df.drop(columns=["plantation", "p", "t", "c", "is_control" , "group", "baseline_timepoint", "wound_closure_baseline", "wound_closure_terminal", "wound_closure_delta", "wound_closure_slope_per_hour"])
y = df["wound_closure_terminal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest

estimator = RandomForestRegressor(n_estimators=100, random_state=42)
estimator.fit(X_train, y_train)

importances = estimator.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# 4. RFE for selection Top-N Features
n_features_to_select = 20
rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42),
          n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

feature_importance_df["Selected_by_RFE"] = rfe.support_


# 5. Threshold feature selection
threshold = 0.01
feature_importance_df["Selected_by_Threshold"] = feature_importance_df["Importance"] > threshold

# Top-N selection
top_n = 10
feature_importance_df["Selected_by_TopN"] = False
feature_importance_df.loc[feature_importance_df.index[:top_n], "Selected_by_TopN"] = True
feature_importance_df.to_csv( OUT_DIR / "rfe_rf_features_eng.csv", index=False)


# Vis RFE-Selected
selected_rfe = feature_importance_df[feature_importance_df["Selected_by_RFE"] == True]

plt.figure(figsize=(10, 6))
plt.barh(selected_rfe["Feature"], selected_rfe["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top Features Selected by RFE")
plt.tight_layout()
plt.savefig(OUT_DIR/ "rfe_rf_feature_importance_plot.png")
plt.show()


# Evalution /accuracy

# Feature sets
features_all = X.columns
features_rfe = feature_importance_df.loc[feature_importance_df["Selected_by_RFE"], "Feature"]
features_thresh = feature_importance_df.loc[feature_importance_df["Selected_by_Threshold"], "Feature"]
features_topn = feature_importance_df.loc[feature_importance_df["Selected_by_TopN"], "Feature"]

# list of results
r2_results = []

def evaluate_models(X_train, X_test, y_train, y_test, feature_set_name, selected_features):
    print(f"\n=== Αξιολόγηση για: {feature_set_name} ===")

    # ----- Random Forest -----
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train[selected_features], y_train)
    y_pred_rf = rf.predict(X_test[selected_features])

    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    print(f"Random Forest R²: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}")
    r2_results.append({
        "Feature_Set": feature_set_name,
        "Model": "RandomForest",
        "R2_Score": r2_rf,
        "RMSE": rmse_rf,
        "MAE": mae_rf
    })

    # ----- MLP Regressor -----
    mlp = MLPRegressor(random_state=42, max_iter=1000, activation='logistic')
    mlp.fit(X_train[selected_features], y_train)
    y_pred_mlp = mlp.predict(X_test[selected_features])

    r2_mlp = r2_score(y_test, y_pred_mlp)
    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)

    print(f"MLP Regressor R²: {r2_mlp:.4f}, RMSE: {rmse_mlp:.4f}, MAE: {mae_mlp:.4f}")
    r2_results.append({
        "Feature_Set": feature_set_name,
        "Model": "MLPRegressor",
        "R2_Score": r2_mlp,
        "RMSE": rmse_mlp,
        "MAE": mae_mlp
    })

evaluate_models(X_train, X_test, y_train, y_test, "All Features", features_all)
evaluate_models(X_train, X_test, y_train, y_test, "RFE Selected", features_rfe)
evaluate_models(X_train, X_test, y_train, y_test, "Threshold > 0.01", features_thresh)
evaluate_models(X_train, X_test, y_train, y_test, "Top-10 Features", features_topn)

# ================================
# 9. save results R², RMSE, MAE
# ================================
r2_df = pd.DataFrame(r2_results)
r2_df.to_csv( OUT_DIR/"rfe_model_scores.csv", index=False)
print("\nThe results (R², RMSE, MAE) have been saved in the 'rfe_model_scores.csv'")

###

rfe_path = OUT_DIR/"rfe_rf_features_eng.csv"
rfe_df = pd.read_csv(rfe_path)

# Start with Feature + Importance from RFE file
out_df = rfe_df[["Feature", "Importance"]].copy()

# Try to pull Gini importances from RF_results.csv (preferred, since it's your RF MDI table)
gini_col = "Gini_Importance"
rf_results_path = "RF_results.csv"
if os.path.exists(rf_results_path):
    rf = pd.read_csv(rf_results_path).rename(columns={"feature": "Feature", "importance": gini_col})
    # Merge by feature name; keep all features from the RFE file
    out_df = out_df.merge(rf[["Feature", gini_col]], on="Feature", how="left")
    # If some features don't match by name, fall back to the RFE 'Importance' column
    out_df[gini_col] = out_df[gini_col].fillna(out_df["Importance"])
else:
    # No RF_results.csv found; use the RFE 'Importance' as Gini (it already came from RF MDI)
    out_df = out_df.rename(columns={"Importance": gini_col})

# Keep only the requested columns, add the always-True flag
out_df = out_df[["Feature", gini_col]].copy()
#out_df["Selected_by_RFE"] = True

# Sort by Gini descending (nice to have)
out_df = out_df.sort_values(by=gini_col, ascending=False)

# Save the "new" CSV
new_csv_path = OUT_DIR/ "rfe_features_eng_new_all_features.csv"  # name it as you like
out_df.to_csv(new_csv_path, index=False)

print(f"Saved '{new_csv_path}' with columns: Feature, {gini_col}, Selected_by_RFE (all True).")


###

rf = pd.read_csv(OUT_DIR/"rfe_features_eng_new_all_features.csv").sort_values(by="Gini_Importance", ascending=False)
rf["cumulative_importance"] = rf["Gini_Importance"].cumsum()

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
plt.savefig(OUT_DIR/"rfe_newplot.png")
plt.show()


# Save the core features (those explaining 80% of importance)
core_features.to_csv(OUT_DIR/ "Rfe_results_eng_80.csv", index=False)
print(f"\n💾 The {len(core_features)} most important features have been saved to 'Rfe_results_eng_80.csv'")
