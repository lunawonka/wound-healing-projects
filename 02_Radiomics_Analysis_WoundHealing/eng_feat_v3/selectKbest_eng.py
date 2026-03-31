import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT_DIR = Path("/data/annapan/alxtu/eng_feat_v3/res_eng/selectKbest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = (
    pd.read_excel("/data/annapan/alxtu/eng_feat_v3/plan_level_engineered_features.xlsx")
      .dropna()
      .sample(frac=1, random_state=42)  
      .reset_index(drop=True)
)

X = df.drop(columns=["plantation", "p", "t", "c", "is_control" , "group", "baseline_timepoint", "wound_closure_baseline", "wound_closure_terminal", "wound_closure_delta", "wound_closure_slope_per_hour"])
y = df["wound_closure_terminal"]

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_score': selector.scores_,
    'P_value': selector.pvalues_
}).sort_values('F_score', ascending=False)

feature_scores.to_csv(OUT_DIR/ "selected_K_best_features.csv", index=False)

top_k = 20
top_features = feature_scores.head(top_k)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_features, x='F_score', y='Feature', hue='Feature', palette='coolwarm')
plt.title(f"Top {top_k} Features by F_regression Score")
plt.xlabel("F-score")
plt.ylabel("Feature")
plt.tight_layout()
plt.grid(True)
plt.savefig(OUT_DIR/"selectKbest_plot.png")
plt.show()
