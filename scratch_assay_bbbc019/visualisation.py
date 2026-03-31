"""
Visualise preprocessed .npy images and masks
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

PROC_DIR      = Path("/data/bbbc_019_clean/processed/images")
PROC_MASK_DIR = Path("/data/bbbc_019_clean/processed/masks")
VIS_DIR       = Path("/data/bbbc_019_clean/results/preprocessing")

# Get only original images (not augmented)
orig_files = sorted([f for f in PROC_DIR.glob("*.npy") 
                     if "_orig" in f.stem])

# Pick 8 random originals to visualise
sample = random.sample(orig_files, min(8, len(orig_files)))

fig, axes = plt.subplots(len(sample), 3, figsize=(12, 4 * len(sample)))

for i, img_path in enumerate(sample):
    mask_path = PROC_MASK_DIR / img_path.name.replace(".npy", "_mask.npy")
    
    img  = np.load(str(img_path))
    mask = np.load(str(mask_path))
    
    wound_pct = mask.mean() * 100
    name      = img_path.stem.replace("_orig", "")
    
    axes[i][0].imshow(img,  cmap="gray")
    axes[i][0].set_title(f"Image\n{name[:40]}", fontsize=8)
    
    axes[i][1].imshow(mask, cmap="gray")
    axes[i][1].set_title(f"Wound mask\n{wound_pct:.1f}% wound area", fontsize=8)
    
    axes[i][2].imshow(img,  cmap="gray", alpha=0.7)
    axes[i][2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[i][2].set_title("Overlay", fontsize=8)
    
    for ax in axes[i]:
        ax.axis("off")

plt.suptitle(
    f"Preprocessed images — 512×512 | BBBC019\n"
    f"Red overlay = wound region",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out = VIS_DIR / "sample_processed_images.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Saved to {out}")


# Also show a few augmented versions of the same image
# to verify augmentation looks correct
print("\nShowing augmentation examples...")
if orig_files:
    base      = orig_files[0].stem.replace("_orig", "")
    aug_files = sorted(PROC_DIR.glob(f"{base}_aug*.npy"))[:3]
    
    if aug_files:
        n   = 1 + len(aug_files)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        
        # Original
        img  = np.load(str(orig_files[0]))
        mask = np.load(str(PROC_MASK_DIR / f"{base}_orig_mask.npy"))
        axes[0][0].imshow(img,  cmap="gray"); axes[0][0].set_title("Original", fontsize=9)
        axes[1][0].imshow(mask, cmap="gray"); axes[1][0].set_title("Mask", fontsize=9)
        
        # Augmented
        for j, aug_path in enumerate(aug_files):
            aug_stem  = aug_path.stem
            aug_mask  = PROC_MASK_DIR / f"{aug_stem}_mask.npy"
            img_aug   = np.load(str(aug_path))
            mask_aug  = np.load(str(aug_mask))
            axes[0][j+1].imshow(img_aug,  cmap="gray")
            axes[0][j+1].set_title(f"Aug {j+1}", fontsize=9)
            axes[1][j+1].imshow(mask_aug, cmap="gray")
            axes[1][j+1].set_title(f"Mask aug {j+1}", fontsize=9)
        
        for row in axes:
            for ax in row:
                ax.axis("off")
        
        plt.suptitle(
            f"Augmentation check — {base[:40]}\n"
            f"Original + 3 augmented versions",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        out2 = VIS_DIR / "augmentation_check.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved to {out2}")