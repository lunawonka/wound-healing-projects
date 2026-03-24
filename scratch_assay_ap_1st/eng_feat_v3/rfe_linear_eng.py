import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.cm as cm

OUT_DIR = Path("/data/annapan/alxtu/eng_feat_v3/res_eng/rfe_linear")
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

estimator = LinearRegression()
n_features_to_select = 20

rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
rfe.fit(X_train, y_train)

importances = np.abs(rfe.estimator_.coef_)
selected_features = X.columns[rfe.support_]

feature_importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("selected features (sorted):")
for index, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.5f}")


colors = cm.Purples(np.linspace(0.9, 0.3, len(feature_importance_df)))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=colors)
plt.gca().invert_yaxis()  
plt.xlabel("Feature Importance")
plt.title("RFE with linear regression - selected features")
plt.tight_layout()
plt.savefig(OUT_DIR/ "rfe_linear_plot.png")
plt.show()

feature_importance_df.to_csv(OUT_DIR/ "features_rfe_linear_anna.csv", index=False)
print("selected features (sorted) are saved at 'features_rfe_linear_anna.csv'")

####------------------------------------------------------------####
#     importances here are "absolute linear regression coefficients"
####------------------------------------------------------------####


# Predict on test set
y_pred = rfe.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R²:  {r2:.4f}")
print(f"RMSE:{rmse:.4f}")
print(f"MAE: {mae:.4f}")

results_linear = pd.DataFrame([{
    "Feature_Set": "RFE_LinearRegression",
    "Model": "LinearRegression",
    "R2_Score": r2,
    "RMSE": rmse,
    "MAE": mae
}])

results_linear.to_csv(OUT_DIR/ "rfe_linear_model_scores.csv", index=False)
print("Saved to rfe_linear_model_scores.csv")
