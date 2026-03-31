"""
Step 1 — MONAI Preprocessing Pipeline
Dataset: BBBC019 (165 paired images + ground truth masks)
Proposal: Month 1-2 — Data curation, annotation, MONAI pipeline deployment
GPU: NVIDIA A100
Resolution: 512x512 (median of dataset)
Augmentation: 3 per image → 165 × 4 = 660 total samples
"""
import os, datetime
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
from monai.transforms import (
    Compose, RandFlip, RandRotate90, RandRotate,
    RandGaussianNoise, RandAdjustContrast, RandZoom
)

# ── CONFIG ────────────────────────────────────────────────
BASE_DIR      = Path("/data/bbbc_019_clean")
IMG_DIR       = BASE_DIR / "images"
MASK_DIR      = BASE_DIR / "masks"
PROC_DIR      = BASE_DIR / "processed/images"
PROC_MASK_DIR = BASE_DIR / "processed/masks"
VIS_DIR       = BASE_DIR / "results/preprocessing"
IMG_SIZE      = 512
AUG_PER_IMAGE = 3        # 165 × 4 = 660 total samples
EXTS          = {'.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg'}

for d in [PROC_DIR, PROC_MASK_DIR, VIS_DIR, BASE_DIR / "results"]:
    os.makedirs(d, exist_ok=True)

# ── GPU VERIFICATION ──────────────────────────────────────
print("="*60)
print("  NVIDIA GPU VERIFICATION — PROPOSAL EVIDENCE")
print("="*60)
print(f"  Timestamp : {datetime.datetime.now()}")
print(f"  GPU       : {torch.cuda.get_device_name(0)}")
print(f"  Memory    : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print(f"  CUDA      : {torch.version.cuda}")
print(f"  PyTorch   : {torch.__version__}")
print("="*60 + "\n")

os.makedirs(BASE_DIR / "results", exist_ok=True)
with open(BASE_DIR / "results/gpu_verification.txt", "w") as f:
    f.write(f"Timestamp : {datetime.datetime.now()}\n")
    f.write(f"GPU       : {torch.cuda.get_device_name(0)}\n")
    f.write(f"Memory    : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB\n")
    f.write(f"CUDA      : {torch.version.cuda}\n")
    f.write(f"PyTorch   : {torch.__version__}\n")


# ── CELL LINE LABEL ───────────────────────────────────────
CELL_LINE_MAP = {
    "SN15":         (0, "DA3"),
    "Init":         (0, "DA3"),
    "Melanoma":     (1, "Melanoma"),
    "MDCK":         (2, "MDCK"),
    "Microfluidic": (2, "MDCK"),
    "Scatter":      (2, "MDCK"),
    "HEK293":       (3, "HEK293T"),
    "TScratch":     (4, "Unknown"),
}

def get_cell_line(stem):
    for prefix, (cl_id, cl_name) in CELL_LINE_MAP.items():
        if stem.startswith(prefix):
            return cl_id, cl_name
    return 0, "DA3"


# ── LOAD + RESIZE HELPERS ─────────────────────────────────
def load_image(path):
    """Load microscopy image → normalised float32 2D grayscale array."""
    try:
        img = tifffile.imread(str(path))
    except Exception:
        img = np.array(Image.open(str(path)).convert("L"))

    # Handle multi-frame TIF or RGB
    if img.ndim == 3 and img.shape[0] <= 4:
        img = img[0]              # first frame
    elif img.ndim == 3:
        img = img.mean(axis=2)    # RGB → grayscale

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def load_mask(path):
    """
    Load BBBC019 ground truth mask.
    White (255) = cell-covered, Black (0) = wound.
    Returns wound mask: wound=1, cells=0.
    """
    m     = np.array(Image.open(str(path)).convert("L"))
    cells = (m > 127).astype(np.float32)
    wound = 1.0 - cells
    return wound


def center_crop_to_square(arr):
    """Center-crop to square using shorter dimension — preserves aspect ratio."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    min_dim = min(h, w)
    top     = (h - min_dim) // 2
    left    = (w - min_dim) // 2
    return arr[top:top + min_dim, left:left + min_dim]


def resize_arr(arr, size):
    """Center-crop to square then resize to target size."""
    arr = center_crop_to_square(arr)
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    return np.array(pil.resize((size, size), Image.BILINEAR)) / 255.0


# ── MONAI AUGMENTATION PIPELINE ───────────────────────────
augment = Compose([
    RandFlip(prob=0.5,  spatial_axis=0),
    RandFlip(prob=0.5,  spatial_axis=1),
    RandRotate90(prob=0.5),
    RandRotate(prob=0.5, range_x=0.35),
    RandGaussianNoise(prob=0.4, std=0.02),
    RandAdjustContrast(prob=0.4, gamma=(0.7, 1.4)),
    RandZoom(prob=0.3,  min_zoom=0.8, max_zoom=1.2),
])


# ── BUILD PAIRED LIST ─────────────────────────────────────
all_images  = sorted([p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in EXTS])
mask_lookup = {
    p.stem.replace("_manual", ""): p
    for p in MASK_DIR.iterdir()
    if p.suffix.lower() in EXTS
}
pairs = [(img, mask_lookup[img.stem])
         for img in all_images
         if img.stem in mask_lookup]

print(f"Paired image+mask files : {len(pairs)}")
print(f"Augmentations per image : {AUG_PER_IMAGE}")
print(f"Total samples           : {len(pairs) * (AUG_PER_IMAGE + 1)}")
print(f"Output resolution       : {IMG_SIZE}×{IMG_SIZE}\n")


# ── MAIN PREPROCESSING LOOP ───────────────────────────────
records = []

for i, (img_path, mask_path) in enumerate(pairs):
    try:
        stem           = img_path.stem
        cl_id, cl_name = get_cell_line(stem)

        # Load, center-crop, resize
        img  = resize_arr(load_image(img_path),  IMG_SIZE)
        mask = resize_arr(load_mask(mask_path),   IMG_SIZE)
        mask = (mask > 0.5).astype(np.float32)

        wound_pct = round(float(mask.mean() * 100), 2)

        # Save original
        ostem = f"{stem}_orig"
        np.save(PROC_DIR      / f"{ostem}.npy",      img)
        np.save(PROC_MASK_DIR / f"{ostem}_mask.npy", mask)
        records.append({
            "stem":           ostem,
            "source":         img_path.name,
            "cell_line_id":   cl_id,
            "cell_line":      cl_name,
            "wound_area_pct": wound_pct,
            "augmented":      False,
        })

        # Save augmented versions
        img_t  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        for k in range(AUG_PER_IMAGE):
            aug_img  = augment(img_t).squeeze().numpy()
            aug_mask = augment(mask_t).squeeze().numpy()
            aug_mask = (aug_mask > 0.5).astype(np.float32)

            astem = f"{stem}_aug{k:02d}"
            np.save(PROC_DIR      / f"{astem}.npy",      aug_img)
            np.save(PROC_MASK_DIR / f"{astem}_mask.npy", aug_mask)
            records.append({
                "stem":           astem,
                "source":         img_path.name,
                "cell_line_id":   cl_id,
                "cell_line":      cl_name,
                "wound_area_pct": round(float(aug_mask.mean() * 100), 2),
                "augmented":      True,
            })

        print(f"[{i+1:3d}/{len(pairs)}] "
              f"{img_path.name:55s} | "
              f"{cl_name:10s} | "
              f"wound={wound_pct:5.1f}%")

    except Exception as e:
        print(f"  ERROR on {img_path.name}: {e}")
        import traceback; traceback.print_exc()


# ── SAVE METADATA CSV ─────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(VIS_DIR / "dataset_metadata.csv", index=False)


# ── SUMMARY ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STEP 1 COMPLETE — MONAI PREPROCESSING")
print(f"{'='*60}")
print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  Resolution     : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Total samples  : {len(df)}")
print(f"  Real images    : {len(df[~df.augmented])}")
print(f"  Augmented      : {len(df[df.augmented])}")
print(f"  Cell lines     :")
for cl, count in df[~df.augmented].cell_line.value_counts().items():
    print(f"    {cl:12s} : {count} images")
print(f"  Mean wound %   : {df.wound_area_pct.mean():.1f}%")
print(f"{'='*60}\n")


# ── VISUALIZATION GRID ────────────────────────────────────
n_show = min(6, len(pairs))
fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))

for i, (img_path, mask_path) in enumerate(pairs[:n_show]):
    try:
        img        = resize_arr(load_image(img_path), IMG_SIZE)
        mask       = resize_arr(load_mask(mask_path), IMG_SIZE)
        _, cl_name = get_cell_line(img_path.stem)

        axes[i][0].imshow(img,  cmap="gray")
        axes[i][0].set_title(f"Raw input — {cl_name}", fontsize=9)

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Ground truth wound mask", fontsize=9)

        axes[i][2].imshow(img,  cmap="gray", alpha=0.7)
        axes[i][2].imshow(mask, cmap="Reds", alpha=0.4)
        axes[i][2].set_title(
            f"Overlay — wound {mask.mean()*100:.1f}%", fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    except Exception as e:
        print(f"  Viz error: {e}")

plt.suptitle(
    f"Step 1: MONAI Preprocessing | BBBC019 | {IMG_SIZE}×{IMG_SIZE}\n"
    f"NVIDIA {torch.cuda.get_device_name(0)} | "
    f"{len(pairs)} images → {len(df)} samples | Proposal Month 1-2",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(VIS_DIR / "preprocessing_grid.png", dpi=150, bbox_inches="tight")
plt.close()


# ── WOUND AREA DISTRIBUTION ───────────────────────────────
orig = df[~df.augmented]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cl in orig.cell_line.unique():
    vals = orig[orig.cell_line == cl].wound_area_pct
    axes[0].hist(vals, bins=12, alpha=0.6, label=cl)
axes[0].set_title("Wound Area Distribution by Cell Line")
axes[0].set_xlabel("Wound Area (%)"); axes[0].set_ylabel("Count")
axes[0].legend()

means = orig.groupby("cell_line")["wound_area_pct"].mean().sort_values()
axes[1].barh(means.index, means.values, color="#76b900", edgecolor="white")
axes[1].set_title("Mean Wound Area by Cell Line")
axes[1].set_xlabel("Mean Wound Area (%)")
for j, v in enumerate(means.values):
    axes[1].text(v + 0.2, j, f"{v:.1f}%", va="center", fontsize=9)

plt.suptitle(
    f"Wound Area Analysis | BBBC019 | NVIDIA A100 | Proposal Month 1-2",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(VIS_DIR / "wound_area_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Saved preprocessing_grid.png")
print(f"✓ Saved wound_area_distribution.png")
print(f"✓ Saved dataset_metadata.csv ({len(df)} rows)")
print(f"✓ Saved gpu_verification.txt")