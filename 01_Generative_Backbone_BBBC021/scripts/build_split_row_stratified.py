from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split

ROOT = Path("/data/annapan/prop/bbbc021")
df = pd.read_csv(ROOT / "bbbc021_moa_resolved.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["moa"],
    random_state=42,
)

split = {
    "train_indices": train_df.index.tolist(),
    "val_indices": val_df.index.tolist(),
}

out_json = ROOT / "split_80_20_row_stratified.json"
with open(out_json, "w") as f:
    json.dump(split, f, indent=2)

print("Saved:", out_json)
print("Train size:", len(train_df))
print("Val size:", len(val_df))

print("\nTrain MoA distribution:")
print(train_df["moa"].value_counts())

print("\nVal MoA distribution:")
print(val_df["moa"].value_counts())
