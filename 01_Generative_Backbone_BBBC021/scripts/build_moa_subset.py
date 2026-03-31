from pathlib import Path
import pandas as pd

ROOT = Path("/data/annapan/prop/bbbc021")

img_csv = ROOT / "BBBC021_v1_image.csv"
moa_csv = ROOT / "BBBC021_v1_moa.csv"

img_df = pd.read_csv(img_csv)
moa_df = pd.read_csv(moa_csv)

df = pd.merge(
    img_df,
    moa_df,
    left_on=["Image_Metadata_Compound", "Image_Metadata_Concentration"],
    right_on=["compound", "concentration"],
    how="inner"
)

out_csv = ROOT / "bbbc021_moa_subset.csv"
df.to_csv(out_csv, index=False)

print("All image rows:", len(img_df))
print("MoA subset rows:", len(df))
print("Unique MoA classes:", df["moa"].nunique())
print("\nMoA distribution:")
print(df["moa"].value_counts())
print(f"\nSaved to: {out_csv}")
