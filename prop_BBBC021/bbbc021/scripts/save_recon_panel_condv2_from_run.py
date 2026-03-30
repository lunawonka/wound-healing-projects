import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_bbbc021_patches import BBBC021PatchDataset
from model.model_cvae_condv2 import ConditionalVAECondV2

ROOT = Path("/data/annapan/prop/bbbc021")
PATCH_METADATA = ROOT / "patches_256_metadata.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denorm(x):
    return (x + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_vis", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "best.pt"

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt.get("config", {})
    moa_to_idx = ckpt["moa_to_idx"]

    dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="val",
        moa_to_idx=moa_to_idx,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.n_vis,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ConditionalVAECondV2(
        num_classes=len(moa_to_idx),
        img_channels=config.get("img_channels", 3),
        latent_dim=config.get("latent_dim", 256),
        cond_dim=config.get("cond_dim", 32),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x, y = next(iter(loader))
    x = x.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)

    with torch.no_grad():
        x_hat, _, _ = model(x, y)

    grid = torch.cat([x, x_hat], dim=0)

    out_path = run_dir / "best_recon_panel.png"
    save_image(
        denorm(grid).clamp(0, 1),
        out_path,
        nrow=args.n_vis,
        pad_value=1.0,
    )

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
    