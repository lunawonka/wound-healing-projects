import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset_bbbc021_patches import BBBC021PatchDataset
from model.model_cgan import Generator, Discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer (crucial for GANs)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_vis_per_class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_sampler", action="store_true")
    return parser.parse_args()


ROOT = Path("/data/annapan/prop/bbbc021")
PATCH_METADATA = ROOT / "patches_256_metadata.csv"
MOA_JSON = ROOT / "moa_to_idx.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm(x):
    return (x + 1.0) / 2.0


def setup_run_dir(root: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path):
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_config(run_dir: Path, config: dict, moa_to_idx: dict):
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "moa_to_idx.json", "w") as f:
        json.dump(moa_to_idx, f, indent=2)


def make_weighted_sampler(dataset):
    labels = [dataset.moa_to_idx[m] for m in dataset.df["moa"].tolist()]
    labels = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=len(dataset.moa_to_idx)).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def save_class_panel(netG, moa_to_idx, epoch, run_dir, device, fixed_z, n_per_class=4):
    netG.eval()
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}
    rows = []
    names = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
            # Use fixed_z so we watch the exact same cells evolve over epochs
            x_gen = netG(fixed_z, y)
            rows.append(x_gen.cpu())
            names.append(idx_to_moa[class_idx])

    all_imgs = torch.cat(rows, dim=0)
    save_image(
        denorm(all_imgs).clamp(0, 1),
        run_dir / "images" / f"class_panel_epoch_{epoch:03d}.png",
        nrow=n_per_class,
        pad_value=1.0,
    )

    with open(run_dir / "images" / f"class_panel_epoch_{epoch:03d}_labels.txt", "w") as f:
        for i, name in enumerate(names):
            f.write(f"row {i}: {name}\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(MOA_JSON, "r") as f:
        moa_to_idx = json.load(f)

    config = vars(args).copy()

    run_dir = setup_run_dir(ROOT, args.run_name)
    logger = setup_logger(run_dir)
    save_config(run_dir, config, moa_to_idx)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Run dir: {run_dir}")

    train_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA, split="train", moa_to_idx=moa_to_idx, augment=True
    )
    val_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA, split="val", moa_to_idx=moa_to_idx, augment=False
    )

    if args.weighted_sampler:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=make_weighted_sampler(train_dataset),
            num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    # Initialize GAN Models
    netG = Generator(num_classes=len(moa_to_idx), latent_dim=args.latent_dim, cond_dim=args.cond_dim).to(DEVICE)
    netD = Discriminator(num_classes=len(moa_to_idx)).to(DEVICE)

    # Loss and Optimizers
    criterion = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Fixed noise for consistent visualization
    fixed_z = torch.randn(args.n_vis_per_class, args.latent_dim, device=DEVICE)

    best_g_loss = float("inf")
    history = []

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        netG.train()
        netD.train()
        
        d_loss_sum = 0.0
        g_loss_sum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

        for x, y in train_bar:
            b_size = x.size(0)
            real_imgs = x.to(DEVICE, non_blocking=True)
            labels = y.to(DEVICE, non_blocking=True)

            # Label smoothing for real images (0.9 instead of 1.0)
            real_targets = torch.full((b_size, 1), 0.9, device=DEVICE)
            fake_targets = torch.full((b_size, 1), 0.0, device=DEVICE)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optD.zero_grad()
            
            # Real batch
            out_real = netD(real_imgs, labels)
            errD_real = criterion(out_real, real_targets)
            
            # Fake batch
            z = torch.randn(b_size, args.latent_dim, device=DEVICE)
            fake_imgs = netG(z, labels)
            out_fake = netD(fake_imgs.detach(), labels) # detach() so we don't backprop into G yet
            errD_fake = criterion(out_fake, fake_targets)
            
            errD = errD_real + errD_fake
            errD.backward()
            optD.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optG.zero_grad()
            
            # G wants D to think the fake images are real (target=1.0)
            out_fake_g = netD(fake_imgs, labels)
            target_for_G = torch.full((b_size, 1), 1.0, device=DEVICE)
            errG = criterion(out_fake_g, target_for_G)
            
            errG.backward()
            optG.step()

            d_loss_sum += errD.item()
            g_loss_sum += errG.item()

            train_bar.set_postfix(D_loss=f"{errD.item():.4f}", G_loss=f"{errG.item():.4f}")

        avg_d_loss = d_loss_sum / len(train_loader)
        avg_g_loss = g_loss_sum / len(train_loader)

        # ---------------------
        # Validation Loop (For tracking only)
        # ---------------------
        netG.eval()
        netD.eval()
        val_d_loss_sum = 0.0
        val_g_loss_sum = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                b_size = x.size(0)
                real_imgs = x.to(DEVICE, non_blocking=True)
                labels = y.to(DEVICE, non_blocking=True)

                real_targets = torch.full((b_size, 1), 0.9, device=DEVICE)
                fake_targets = torch.full((b_size, 1), 0.0, device=DEVICE)

                out_real = netD(real_imgs, labels)
                errD_real = criterion(out_real, real_targets)

                z = torch.randn(b_size, args.latent_dim, device=DEVICE)
                fake_imgs = netG(z, labels)
                
                out_fake = netD(fake_imgs, labels)
                errD_fake = criterion(out_fake, fake_targets)
                val_d_loss_sum += (errD_real + errD_fake).item()

                target_for_G = torch.full((b_size, 1), 1.0, device=DEVICE)
                val_g_loss_sum += criterion(out_fake, target_for_G).item()

        val_d_loss = val_d_loss_sum / len(val_loader)
        val_g_loss = val_g_loss_sum / len(val_loader)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train D_loss={avg_d_loss:.4f} G_loss={avg_g_loss:.4f} | "
            f"Val D_loss={val_d_loss:.4f} G_loss={val_g_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_D_loss": avg_d_loss,
            "train_G_loss": avg_g_loss,
            "val_D_loss": val_d_loss,
            "val_G_loss": val_g_loss,
            "epoch_time_sec": epoch_time,
        })

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_class_panel(netG, moa_to_idx, epoch, run_dir, DEVICE, fixed_z, n_per_class=args.n_vis_per_class)

        ckpt = {
            "epoch": epoch,
            "netG_state": netG.state_dict(),
            "netD_state": netD.state_dict(),
            "optG_state": optG.state_dict(),
            "optD_state": optD.state_dict(),
            "moa_to_idx": moa_to_idx,
            "config": config,
        }

        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")

        # Note: "Best" in GANs is subjective without FID. We save based on Val G_Loss here as a proxy.
        if val_g_loss < best_g_loss:
            best_g_loss = val_g_loss
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            logger.info(f"Saved best model at epoch {epoch}")

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
