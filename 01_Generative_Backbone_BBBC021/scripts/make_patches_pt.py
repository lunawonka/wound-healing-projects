from pathlib import Path
import pandas as pd
import json
import torch
import tifffile as tiff
import numpy as np

ROOT = Path("/data/annapan/prop/bbbc021")
OUT_DIR = ROOT / "patches_256_pt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

resolved_csv = ROOT / "bbbc021_moa_resolved.csv"
split_json = ROOT / "split_80_20_row_stratified.json"

PATCH = 256
STRIDE = 256

df = pd.read_csv(resolved_csv)

with open(split_json, "r") as f:
    split_data = json.load(f)

train_set = set(split_data["train_indices"])
val_set = set(split_data["val_indices"])

def norm(x):
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max() - x.min()
    if denom > 0:
        x = x / denom
    return x

records = []

for idx, row in df.iterrows():
    dapi = norm(tiff.imread(row["path_dapi"]))
    actin = norm(tiff.imread(row["path_actin"]))
    tubulin = norm(tiff.imread(row["path_tubulin"]))

    img = np.stack([dapi, actin, tubulin], axis=0)  # [3, H, W]
    _, H, W = img.shape

    split = "train" if idx in train_set else "val" if idx in val_set else None
    if split is None:
        continue

    patch_id = 0
    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            patch = img[:, y:y+PATCH, x:x+PATCH]

            # skip almost-empty patches
            if patch.mean() < 0.02:
                continue

            patch_tensor = torch.from_numpy(patch).float()

            subdir = OUT_DIR / split / row["moa"].replace("/", "_")
            subdir.mkdir(parents=True, exist_ok=True)

            fname = f"img{idx:05d}_patch{patch_id:02d}.pt"
            fpath = subdir / fname
            torch.save(patch_tensor, fpath)

            records.append({
                "source_index": idx,
                "split": split,
                "moa": row["moa"],
                "compound": row["Image_Metadata_Compound"],
                "concentration": row["Image_Metadata_Concentration"],
                "path_pt": str(fpath),
                "patch_id": patch_id,
                "x": x,
                "y": y,
            })

            patch_id += 1

meta_df = pd.DataFrame(records)
meta_df.to_csv(ROOT / "patches_256_metadata.csv", index=False)

print("Saved patches:", len(meta_df))
print(meta_df["split"].value_counts())
print(meta_df["moa"].value_counts())
print("Metadata:", ROOT / "patches_256_metadata.csv")
