from pathlib import Path
import pandas as pd

ROOT = Path("/data/annapan/prop/bbbc021")
EXTRACTED = ROOT / "extracted"

records = []

for p in EXTRACTED.rglob("*.tif"):
    records.append({
        "filename": p.name,
        "path": str(p.resolve())
    })

for p in EXTRACTED.rglob("*.TIF"):
    records.append({
        "filename": p.name,
        "path": str(p.resolve())
    })

df = pd.DataFrame(records).drop_duplicates(subset=["filename"])
out_csv = ROOT / "file_index.csv"
df.to_csv(out_csv, index=False)

print("Indexed files:", len(df))
print("Saved:", out_csv)
