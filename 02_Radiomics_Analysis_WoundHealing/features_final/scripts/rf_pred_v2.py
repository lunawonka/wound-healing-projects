import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load and clean data
df = pd.read_csv("/data/annapan/alxtu/features_final/data_for_training_testing_stripped.csv").dropna()
df = df[df["timepoint"] != 0]

# treatment label: 1 = successful treatment, 0 = unsuccessful treatment (t = 0 or 2)
df["treatment"] = df["t"].apply(lambda x: 1 if x == 1 else 0)

# 2. Define features X and target y
drop_cols = [
    "image_id", "largest_region_area", "plantation", "initial_area",
    "wound_openess", "wound_closure", "patch", "p", "t", "c",
    "is_control", "treatment"  # treatment is the target, so drop from X
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["treatment"]

print("X shape:", X.shape)
print("y value counts (0=unsuccessful, 1=successful):\n", y.value_counts())

# 3. Define model and cross-validation
clf = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# 4. Cross-validated performance
cv_results = cross_validate(
    clf,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

acc_mean = cv_results["test_accuracy"].mean()
acc_std = cv_results["test_accuracy"].std()

f1_mean = cv_results["test_f1"].mean()
f1_std = cv_results["test_f1"].std()

auc_mean = cv_results["test_roc_auc"].mean()
auc_std = cv_results["test_roc_auc"].std()

print(f"\nCross-validated Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
print(f"Cross-validated F1-score: {f1_mean:.3f} ± {f1_std:.3f}")
print(f"Cross-validated ROC-AUC:  {auc_mean:.3f} ± {auc_std:.3f}")

# 5. Cross-validated predictions for confusion matrix and report
y_pred_cv = cross_val_predict(
    clf,
    X,
    y,
    cv=cv,
    method="predict",
    n_jobs=-1
)

cm = confusion_matrix(y, y_pred_cv, labels=[0, 1])
print("\nConfusion Matrix (rows=true, cols=pred) [0=unsuccessful, 1=successful]:\n", cm)

print(
    "\nClassification Report (0=unsuccessful treatment, 1=successful treatment):\n",
    classification_report(
        y,
        y_pred_cv,
        target_names=["unsuccessful treatment", "successful treatment"]
    )
)

# 6. Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred: Unsuccessful", "Pred: Successful"],
    yticklabels=["True: Unsuccessful", "True: Successful"]
)
plt.title("Random Forest – Confusion Matrix (5-fold CV)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

output_dir = "/data/annapan/alxtu/plots"
os.makedirs(output_dir, exist_ok=True)
cm_path = os.path.join(output_dir, "rf_confusion_matrix_cv_v2.png")

plt.tight_layout()
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"✅ Confusion matrix saved at: {cm_path}")

# 7. Fit on full data to get feature importances
clf.fit(X, y)

feature_importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
      .sort_values(by="Importance", ascending=False)
)

print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15))

top_features = feature_importance_df.head(15).copy()
top_features["Importance (%)"] = top_features["Importance"] * 100

total_top15 = top_features["Importance"].sum() * 100
print(f"\n📊 The top 15 features account for {total_top15:.2f}% of the total model importance.")

plt.figure(figsize=(9, 6))
sns.barplot(
    data=top_features,
    x="Importance (%)",
    y="Feature",
    hue="Feature",
    palette="Blues_d",
    legend=False
)
plt.title("Top 15 Most Important Features (Random Forest)", fontsize=14)
plt.xlabel("Importance (%)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()

fi_path = os.path.join(output_dir, "rf_feature_importances_top15_v2.png")
plt.savefig(fi_path, dpi=300)
plt.show()

print(f"✅ Feature importance plot saved at: {fi_path}")
