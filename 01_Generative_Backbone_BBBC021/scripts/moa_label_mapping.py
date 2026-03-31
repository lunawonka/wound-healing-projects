import pandas as pd
import json

df = pd.read_csv("/data/annapan/prop/bbbc021/patches_256_metadata.csv")
moas = sorted(df["moa"].unique().tolist())
moa_to_idx = {m: i for i, m in enumerate(moas)}

with open("/data/annapan/prop/bbbc021/moa_to_idx.json", "w") as f:
    json.dump(moa_to_idx, f, indent=2)

print(moa_to_idx)
