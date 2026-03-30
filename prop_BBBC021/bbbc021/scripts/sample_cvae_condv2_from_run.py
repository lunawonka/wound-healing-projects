import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from model.model_cvae_condv2 import ConditionalVAECondV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denorm(x):
    return (x + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_per_class", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "best.pt"

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt.get("config", {})
    moa_to_idx = ckpt["moa_to_idx"]
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    model = ConditionalVAECondV2(
        num_classes=len(moa_to_idx),
        img_channels=config.get("img_channels", 3),
        latent_dim=config.get("latent_dim", 128),
        cond_dim=config.get("cond_dim", 32),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    labels = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((args.n_per_class,), class_idx, dtype=torch.long, device=DEVICE)
            x = model.sample(y, device=DEVICE)
            rows.append(x.cpu())
            labels.append(idx_to_moa[class_idx])

    imgs = torch.cat(rows, dim=0)
    out_path = run_dir / "best_class_panel.png"

    save_image(
        denorm(imgs).clamp(0, 1),
        out_path,
        nrow=args.n_per_class,
        pad_value=1.0,
    )

    with open(run_dir / "best_class_panel_labels.txt", "w") as f:
        for i, name in enumerate(labels):
            f.write(f"row {i}: {name}\n")

    print("Saved:", out_path)
    print("Saved labels:", run_dir / "best_class_panel_labels.txt")


if __name__ == "__main__":
    main()


    #python sample_cvae_condv2_from_run.py --run_dir /data/annapan/prop/bbbc021/runs/20260320_104638_cvae_condv2_patch256_ld128_beta5e-4_bs32_augfliprot_e80