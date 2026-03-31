import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from model.model_cvae import ConditionalVAE


ROOT = Path("/data/annapan/prop/bbbc021")
CKPT_PATH = ROOT / "runs" / "cvae_v1" / "best.pt"
OUT_PATH = ROOT / "runs" / "cvae_v1" / "final_samples.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denorm(x):
    return (x + 1.0) / 2.0


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    moa_to_idx = ckpt["moa_to_idx"]
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    model = ConditionalVAE(
        num_classes=len(moa_to_idx),
        img_channels=3,
        latent_dim=128,
        cond_dim=32,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    imgs = []
    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((8,), class_idx, dtype=torch.long, device=DEVICE)
            x = model.sample(y, device=DEVICE)
            imgs.append(x.cpu())

    imgs = torch.cat(imgs, dim=0)

    save_image(
        denorm(imgs).clamp(0, 1),
        OUT_PATH,
        nrow=8
    )

    print("Saved samples to:", OUT_PATH)
    print("Class order:")
    for i in range(len(moa_to_idx)):
        print(i, idx_to_moa[i])


if __name__ == "__main__":
    main()
