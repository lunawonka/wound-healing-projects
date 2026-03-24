import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

df = pd.read_excel("/data/annapan/alxtu/eng_feat_v3/plan_level_engineered_features.xlsx").dropna()
#df = df[df["timepoint"] != 0]
df["therapy"] = df["t"].apply(lambda x: 1 if x == 1 else 0)

X = df.drop(columns=["plantation", "p", "t", "c", "is_control" , "group", "baseline_timepoint", "wound_closure_baseline", "wound_closure_terminal", "wound_closure_delta", "wound_closure_slope_per_hour", "therapy"])
y = df["therapy"] #y = df["wound_closure_terminal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

t_test = df.loc[X_test.index, "t"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (positive=good therapy):", f1_score(y_test, y_pred))
print("Confusion Matrix (rows=true, cols=pred) [bad(0), good(1)]:\n", confusion_matrix(y_test, y_pred, labels=[0,1]))
print("\nClassification Report (0=bad therapy, 1=good therapy):\n",
      classification_report(y_test, y_pred, target_names=["bad therapy","good therapy"]))

bad_mask = t_test.isin([0, 2])
good_mask = t_test.eq(1)

correct_bad = ((bad_mask) & (y_pred == 0)).sum()
total_bad = bad_mask.sum()

correct_good = ((good_mask) & (y_pred == 1)).sum()
total_good = good_mask.sum()

print(f"\nCorrect 'bad' predictions (t in {{0,2}} predicted 0): {correct_bad} / {total_bad}")
print(f"Correct 'good' predictions (t == 1 predicted 1): {correct_good} / {total_good}")

feature_importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
      .sort_values(by="Importance", ascending=False)
)
print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15))

pred_text = np.where(y_pred == 1, "good therapy", "bad therapy")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Pred: Bad", "Pred: Good"],
    yticklabels=["True: Bad", "True: Good"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

output_dir = "/data/annapan/alxtu/eng_feat_v3/res_eng/preds_rf"  
os.makedirs(output_dir, exist_ok=True)     

output_path = os.path.join(output_dir, "confusion_matrix.png")

plt.tight_layout()
plt.savefig(output_path, dpi=300)  
plt.show()

print(f"Confusion matrix saved at: {output_path}")

top_features = feature_importance_df.head(15).copy()
top_features["Importance (%)"] = top_features["Importance"] * 100

total_top15 = top_features["Importance"].sum() * 100
print(f"Τα 15 πιο σημαντικά χαρακτηριστικά εξηγούν συνολικά το {total_top15:.2f}% της σημασίας του μοντέλου.")

plt.figure(figsize=(9, 6))
sns.barplot(
    data=top_features,
    x="Importance (%)",
    y="Feature",
    hue="Feature",
    palette="coolwarm",
    legend=False
)
plt.title("Top 15 Most Important Features (Random Forest)", fontsize=14)
plt.xlabel("Importance (%)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()

output_path = os.path.join(output_dir, "feature_importances.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Feature importance plot saved at: {output_path}")
