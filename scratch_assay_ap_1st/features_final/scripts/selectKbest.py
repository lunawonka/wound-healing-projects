import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

csv_path = "/data/annapan/alxtu/features_final/data_for_training_testing_stripped_reduced.csv"  
df = pd.read_csv(csv_path)

y = df.iloc[:, 1]  # wound_closure
X = df.iloc[:, 2:]  # exclude wound_openness and wound_closure

X = X.dropna()
y = y[X.index]

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_score': selector.scores_,
    'P_value': selector.pvalues_
}).sort_values('F_score', ascending=False)

feature_scores.to_csv("selected_K_best_features.csv", index=False)

top_k = 20
top_features = feature_scores.head(top_k)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_features, x='F_score', y='Feature', hue='Feature', palette='coolwarm')
plt.title(f"Top {top_k} Features by F_regression Score")
plt.xlabel("F-score")
plt.ylabel("Feature")
plt.tight_layout()
plt.grid(True)
plt.savefig("selectKbest_plot.png")
plt.show()
