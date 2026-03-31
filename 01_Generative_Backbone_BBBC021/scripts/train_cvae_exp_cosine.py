import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset_bbbc021_patches import BBBC021PatchDataset
from model.model_cvae import ConditionalVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--beta", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_vis_recon", type=int, default=6)
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


def vae_loss(x_hat, x, mu, logvar, beta=5e-4):
    recon = F.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


def save_reconstructions(model, loader, epoch, run_dir, device, n_vis=6):
    model.eval()
    x, y = next(iter(loader))
    x = x[:n_vis].to(device)
    y = y[:n_vis].to(device)

    with torch.no_grad():
        x_hat, _, _ = model(x, y)

    grid = torch.cat([x, x_hat], dim=0)
    save_image(
        denorm(grid).clamp(0, 1),
        run_dir / "images" / f"recon_epoch_{epoch:03d}.png",
        nrow=n_vis,
    )


def save_class_panel(model, moa_to_idx, epoch, run_dir, device, n_per_class=4):
    model.eval()
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}
    rows = []
    names = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
            x_gen = model.sample(y, device=device)
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
    config["scheduler"] = "CosineAnnealingLR"

    run_dir = setup_run_dir(ROOT, args.run_name)
    logger = setup_logger(run_dir)
    save_config(run_dir, config, moa_to_idx)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Run dir: {run_dir}")

    train_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="train",
        moa_to_idx=moa_to_idx,
        augment=True,
    )
    val_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="val",
        moa_to_idx=moa_to_idx,
        augment=False,
    )

    if args.weighted_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=make_weighted_sampler(train_dataset),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ConditionalVAE(
        num_classes=len(moa_to_idx),
        img_channels=3,
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_val = float("inf")
    history = []

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

        for x, y in train_bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(x, y)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += recon.item()
            train_kl_sum += kl.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = train_loss_sum / len(train_loader)
        train_recon = train_recon_sum / len(train_loader)
        train_kl = train_kl_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False)

        with torch.no_grad():
            for x, y in val_bar:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                x_hat, mu, logvar = model(x, y)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=args.beta)

                val_loss_sum += loss.item()
                val_recon_sum += recon.item()
                val_kl_sum += kl.item()

                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon.item():.4f}",
                    kl=f"{kl.item():.4f}",
                )

        val_loss = val_loss_sum / len(val_loader)
        val_recon = val_recon_sum / len(val_loader)
        val_kl = val_kl_sum / len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val_loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_kl": train_kl,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
            "epoch_time_sec": epoch_time,
        })

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_reconstructions(model, val_loader, epoch, run_dir, DEVICE, n_vis=args.n_vis_recon)
        save_class_panel(model, moa_to_idx, epoch, run_dir, DEVICE, n_per_class=args.n_vis_per_class)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "moa_to_idx": moa_to_idx,
            "config": config,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
        }

        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            logger.info(f"Saved best model at epoch {epoch}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
    