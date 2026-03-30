from pathlib import Path
import pandas as pd

ROOT = Path("/data/annapan/prop/bbbc021")

moa_df = pd.read_csv(ROOT / "bbbc021_moa_subset.csv")
file_index = pd.read_csv(ROOT / "file_index.csv")

fname_to_path = dict(zip(file_index["filename"], file_index["path"]))

rows = []
missing = 0

for _, row in moa_df.iterrows():
    dapi_name = row["Image_FileName_DAPI"]
    actin_name = row["Image_FileName_Actin"]
    tub_name = row["Image_FileName_Tubulin"]

    dapi_path = fname_to_path.get(dapi_name)
    actin_path = fname_to_path.get(actin_name)
    tub_path = fname_to_path.get(tub_name)

    if dapi_path is None or actin_path is None or tub_path is None:
        missing += 1
        continue

    rec = row.to_dict()
    rec["path_dapi"] = dapi_path
    rec["path_actin"] = actin_path
    rec["path_tubulin"] = tub_path
    rows.append(rec)

resolved_df = pd.DataFrame(rows)
out_csv = ROOT / "bbbc021_moa_resolved.csv"
resolved_df.to_csv(out_csv, index=False)

print("Resolved rows:", len(resolved_df))
print("Missing rows:", missing)
print("Saved:", out_csv)
