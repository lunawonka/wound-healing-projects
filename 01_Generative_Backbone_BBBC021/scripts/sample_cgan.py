import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

# Corrected import to match the class name we used in model_cgan.py
from model.model_cgan import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denorm(x):
    return (x + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run folder (e.g., runs/20260320_..._cgan)")
    parser.add_argument("--n_per_class", type=int, default=4, help="Number of images to generate per MoA class")
    parser.add_argument("--ckpt_name", type=str, default="last.pt", help="Which checkpoint to load (e.g., last.pt or best.pt)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / args.ckpt_name

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    config = ckpt.get("config", {})
    moa_to_idx = ckpt["moa_to_idx"]
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    latent_dim = config.get("latent_dim", 128)
    cond_dim = config.get("cond_dim", 128)

    # Initialize the Generator using the correct class name
    G = Generator(
        num_classes=len(moa_to_idx),
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        img_channels=3,
    ).to(DEVICE)

    # Corrected the key to look for "netG_state" which is what train_cgan.py saves
    if "netG_state" in ckpt:
        G.load_state_dict(ckpt["netG_state"])
    elif "generator_state" in ckpt:
        G.load_state_dict(ckpt["generator_state"])
    elif "G_state" in ckpt:
        G.load_state_dict(ckpt["G_state"])
    else:
        raise KeyError("Could not find generator weights in checkpoint. Available keys: ", ckpt.keys())

    G.eval()

    rows = []
    labels = []

    print("Generating samples...")
    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            # Create labels and noise for this specific class
            y = torch.full((args.n_per_class,), class_idx, dtype=torch.long, device=DEVICE)
            z = torch.randn(args.n_per_class, latent_dim, device=DEVICE)
            
            # Generate images
            x_gen = G(z, y)
            
            rows.append(x_gen.cpu())
            labels.append(idx_to_moa[class_idx])

    # Concatenate all rows into a single tensor grid
    imgs = torch.cat(rows, dim=0)
    
    # Define output paths
    base_name = args.ckpt_name.replace('.pt', '')
    out_img_path = run_dir / f"cgan_{base_name}_class_panel.png"
    out_txt_path = run_dir / f"cgan_{base_name}_class_panel_labels.txt"

    # Save the image grid
    save_image(
        denorm(imgs).clamp(0, 1),
        out_img_path,
        nrow=args.n_per_class,
        pad_value=1.0,
    )

    # Save the corresponding labels
    with open(out_txt_path, "w") as f:
        for i, name in enumerate(labels):
            f.write(f"row {i}: {name}\n")

    print(f"✅ Saved image panel to: {out_img_path}")
    print(f"✅ Saved labels to: {out_txt_path}")


if __name__ == "__main__":
    main()
    