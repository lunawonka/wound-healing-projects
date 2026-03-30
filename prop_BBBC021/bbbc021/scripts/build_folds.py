from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import GroupKFold

ROOT = Path("/data/annapan/prop/bbbc021")
df = pd.read_csv(ROOT / "bbbc021_moa_resolved.csv")

# Group by compound to avoid leakage
groups = df["Image_Metadata_Compound"].astype(str).values

gkf = GroupKFold(n_splits=5)

folds = {}

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
    folds[f"fold_{fold_idx}"] = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist()
    }

out_json = ROOT / "folds_by_compound.json"
with open(out_json, "w") as f:
    json.dump(folds, f, indent=2)

print("Saved:", out_json)
for k, v in folds.items():
    print(k, len(v["train_indices"]), len(v["val_indices"]))
    