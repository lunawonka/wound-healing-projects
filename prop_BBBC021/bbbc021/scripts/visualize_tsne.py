import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset_bbbc021_patches import BBBC021PatchDataset
from model.model_cvae import ConditionalVAE

# =========================
# Setup Paths
# =========================
ROOT = Path("/data/annapan/prop/bbbc021")
RUN_DIR = ROOT / "runs" / "20260320_092756_cvae_patch256_ld128_beta5e-4_bs32_augfliprot_e80"
CKPT_PATH = RUN_DIR / "checkpoints" / "best.pt"
PATCH_METADATA = ROOT / "patches_256_metadata.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt.get("config", {})
    moa_to_idx = ckpt["moa_to_idx"]
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    # Robust fallback values
    num_classes = len(moa_to_idx)
    img_channels = config.get("img_channels", 3)
    latent_dim = config.get("latent_dim", 128)
    cond_dim = config.get("cond_dim", 32)
    beta = config.get("beta", "unknown")

    print(f"num_classes={num_classes}, img_channels={img_channels}, latent_dim={latent_dim}, cond_dim={cond_dim}")

    val_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="val",
        moa_to_idx=moa_to_idx,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = ConditionalVAE(
        num_classes=num_classes,
        img_channels=img_channels,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Extracting latent mean vectors...")
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            mu, _ = model.encode(x, y)
            all_mu.append(mu.cpu().numpy())
            all_labels.extend([idx_to_moa[int(lbl)] for lbl in y.cpu()])

    all_mu = np.concatenate(all_mu, axis=0)
    print("Latent matrix shape:", all_mu.shape)

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    )
    latent_2d = tsne.fit_transform(all_mu)

    # Map string labels to integer colors
    class_names = sorted(set(all_labels))
    class_to_int = {name: i for i, name in enumerate(class_names)}
    color_ids = np.array([class_to_int[name] for name in all_labels])

    print("Plotting...")
    plt.figure(figsize=(14, 11))
    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=color_ids,
        s=10,
        alpha=0.65,
        cmap="tab20",
    )

    # Build legend manually
    handles = []
    for class_name in class_names:
        idx = class_to_int[class_name]
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=class_name,
                markerfacecolor=plt.cm.tab20(idx / max(1, len(class_names) - 1)),
                markersize=7,
            )
        )

    plt.title(f"t-SNE of cVAE Latent Space (beta={beta})", fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=9)
    plt.tight_layout()

    out_path = RUN_DIR / "images" / "latent_tsne_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
    