import torch
from pathlib import Path
from torchvision.utils import save_image

from model.model_cvae import ConditionalVAE

RUN_DIR = Path("/data/annapan/prop/bbbc021/runs/20260319_144803_cvae_patch256_ld128_beta1e-3_bs32_augfliprot_e60")
CKPT_PATH = RUN_DIR / "checkpoints" / "best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denorm(x):
    return (x + 1.0) / 2.0


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    moa_to_idx = ckpt["moa_to_idx"]
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}
    config = ckpt["config"]

    model = ConditionalVAE(
        num_classes=len(moa_to_idx),
        img_channels=3,
        latent_dim=config["latent_dim"],
        cond_dim=config["cond_dim"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_per_class = 4
    rows = []
    labels = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=DEVICE)
            x = model.sample(y, device=DEVICE)
            rows.append(x.cpu())
            labels.append(idx_to_moa[class_idx])

    imgs = torch.cat(rows, dim=0)
    out_path = RUN_DIR / "poster_friendly_samples.png"

    save_image(
        denorm(imgs).clamp(0, 1),
        out_path,
        nrow=n_per_class,
        pad_value=1.0,
    )

    with open(RUN_DIR / "poster_friendly_samples_labels.txt", "w") as f:
        for i, name in enumerate(labels):
            f.write(f"row {i}: {name}\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
