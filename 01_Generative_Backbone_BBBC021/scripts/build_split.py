from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("/data/annapan/prop/bbbc021")
df = pd.read_csv(ROOT / "bbbc021_moa_resolved.csv")

X = df.index.values
y = df["moa"].astype(str).values
groups = df["Image_Metadata_Compound"].astype(str).values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf.split(X, y, groups))

split = {
    "train_indices": train_idx.tolist(),
    "val_indices": val_idx.tolist()
}

out_json = ROOT / "split_80_20_stratified_grouped.json"
with open(out_json, "w") as f:
    json.dump(split, f, indent=2)

print("Saved:", out_json)
print("Train size:", len(train_idx))
print("Val size:", len(val_idx))

print("\nTrain MoA distribution:")
print(df.iloc[train_idx]["moa"].value_counts())

print("\nVal MoA distribution:")
print(df.iloc[val_idx]["moa"].value_counts())

train_compounds = set(df.iloc[train_idx]["Image_Metadata_Compound"].astype(str))
val_compounds = set(df.iloc[val_idx]["Image_Metadata_Compound"].astype(str))
print("\nCompound overlap:", len(train_compounds & val_compounds))
