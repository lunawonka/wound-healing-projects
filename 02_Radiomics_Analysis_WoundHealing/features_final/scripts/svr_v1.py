import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

df = pd.read_csv("/data/annapan/alxtu/features_final/data_for_training_testing_stripped_reduced.csv")

print("initial shape:", df.shape)
df = df.dropna()  # drop NaNs
print("without NaNs:", df.shape)

X = df.iloc[:, 2:]  # exclude wound_openness and wound_closure
X = X.dropna()
#separate Features & Target
#X = df.drop("wound_closure", axis=1)
y = df["wound_closure"]

feature_names = X.columns  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline with StandardScaler + SVR (RBF kernel)
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

# Training SVR
svr_pipeline.fit(X_train, y_train)

#Eval in the test set
r2_test = svr_pipeline.score(X_test, y_test)
print(f"R² στο test set: {r2_test:.4f}")

# Permutation Importance for SVR
print("computation of permutation importance.")
result = permutation_importance(
    svr_pipeline,
    X_test,
    y_test,
    n_repeats=100,        
    random_state=42,
    n_jobs=-1
)

importances_mean = result.importances_mean
importances_std = result.importances_std

#DataFrame with the results
feature_importances_svr = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": importances_mean,
    "importance_std": importances_std
}).sort_values(by="importance_mean", ascending=False)

print("Top features (SVR permutation importance):")
print(feature_importances_svr.head(20))

#Top 20 Features 
top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(
    x="importance_mean",
    y="feature",
    data=feature_importances_svr.head(top_n)
)
plt.title(f"Top {top_n} most important features for wound_closure (SVR – permutation importance)")
plt.xlabel("Mean importance")
plt.ylabel("Feature")
plt.tight_layout()
# Save high-resolution figure for the paper
plt.savefig("/data/annapan/alxtu/features_final/svr_permutation_importance_top20.png", dpi=300, bbox_inches="tight")
plt.show()

# save_results
feature_importances_svr.to_csv("/data/annapan/alxtu/features_final/SVR_feature_importances.csv", index=False)
print("SVR permutation importances are saved at 'SVR_feature_importances.csv'")

import shap

# 1. Background data for SHAP (small subset for performance)
background_size = 100
X_background = X_train.sample(background_size, random_state=42)

# 2. Define a pure prediction function that uses the trained pipeline
def svr_predict(X_array):
    # X_array will come as a numpy array from SHAP; convert to DataFrame with correct columns
    X_df = pd.DataFrame(X_array, columns=X_train.columns)
    return svr_pipeline.predict(X_df)

# 3. Create KernelExplainer using the prediction function, not the pipeline object
explainer = shap.KernelExplainer(
    svr_predict,
    X_background.to_numpy(),   # SHAP background as numpy array
)

# 4. Take a manageable subset of the test data for SHAP
test_size = 100
X_test_sample = X_test.sample(test_size, random_state=42)

print("Computing SHAP values... this may take several minutes.")
shap_values = explainer.shap_values(X_test_sample.to_numpy())

# 5. SHAP summary plot (global importance)
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=X_train.columns,
    show=False
)
plt.tight_layout()
plt.savefig("/data/annapan/alxtu/features_final/svr_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. SHAP beeswarm plot (distribution of effects)
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=X_train.columns,
    plot_type="dot",
    show=False
)
plt.tight_layout()
plt.savefig("/data/annapan/alxtu/features_final/svr_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP plots saved as 'svr_shap_summary.png' and 'svr_shap_beeswarm.png'")
