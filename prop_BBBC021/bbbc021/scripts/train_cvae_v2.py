import os
import json
import time
import math
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


# =========================
# Config
# =========================
ROOT = Path("/data/annapan/prop/bbbc021")
PATCH_METADATA = ROOT / "patches_256_metadata.csv"
MOA_JSON = ROOT / "moa_to_idx.json"

CONFIG = {
    "run_name": "cvae_patch256_ld128_beta1e-3_bs32_augfliprot_e60",
    "batch_size": 32,
    "num_workers": 8,
    "latent_dim": 128,
    "cond_dim": 32,
    "lr": 1e-4,
    "epochs": 60,
    "beta": 1e-3,
    "img_channels": 3,
    "num_classes": 13,
    "n_vis_recon": 6,
    "n_vis_per_class": 4,
    "seed": 42,
    "use_weighted_sampler": True,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utilities
# =========================
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

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def vae_loss(x_hat, x, mu, logvar, beta=1e-3):
    recon = F.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


# =========================
# Visualization helpers
# =========================
def save_reconstructions(model, loader, epoch, run_dir, device, n_vis=6):
    model.eval()
    x, y = next(iter(loader))
    x = x[:n_vis].to(device)
    y = y[:n_vis].to(device)

    with torch.no_grad():
        x_hat, _, _ = model(x, y)

    # alternating rows: originals then reconstructions
    grid = torch.cat([x, x_hat], dim=0)
    save_image(
        denorm(grid).clamp(0, 1),
        run_dir / "images" / f"recon_epoch_{epoch:03d}.png",
        nrow=n_vis,
    )


def save_class_samples_readable(model, moa_to_idx, epoch, run_dir, device, n_per_class=4):
    model.eval()
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    rows = []
    class_names = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
            x_gen = model.sample(y, device=device)
            rows.append(x_gen.cpu())
            class_names.append(idx_to_moa[class_idx])

    # concatenate all class rows
    all_imgs = torch.cat(rows, dim=0)

    save_image(
        denorm(all_imgs).clamp(0, 1),
        run_dir / "images" / f"class_samples_epoch_{epoch:03d}.png",
        nrow=n_per_class,
        pad_value=1.0,
    )

    with open(run_dir / "images" / f"class_samples_epoch_{epoch:03d}_labels.txt", "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"row {i}: {name}\n")


def save_fixed_class_panel(model, moa_to_idx, run_dir, device, ckpt_name="best"):
    model.eval()
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}
    n_per_class = 4

    rows = []
    class_names = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
            x_gen = model.sample(y, device=device)
            rows.append(x_gen.cpu())
            class_names.append(idx_to_moa[class_idx])

    all_imgs = torch.cat(rows, dim=0)
    out_path = run_dir / f"{ckpt_name}_class_panel.png"

    save_image(
        denorm(all_imgs).clamp(0, 1),
        out_path,
        nrow=n_per_class,
        pad_value=1.0,
    )

    with open(run_dir / f"{ckpt_name}_class_panel_labels.txt", "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"row {i}: {name}\n")


# =========================
# Main
# =========================
def main():
    set_seed(CONFIG["seed"])

    with open(MOA_JSON, "r") as f:
        moa_to_idx = json.load(f)

    run_dir = setup_run_dir(ROOT, CONFIG["run_name"])
    logger = setup_logger(run_dir)
    save_config(run_dir, CONFIG, moa_to_idx)

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

    if CONFIG["use_weighted_sampler"]:
        train_sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if CONFIG["num_workers"] > 0 else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if CONFIG["num_workers"] > 0 else False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True if CONFIG["num_workers"] > 0 else False,
    )

    model = ConditionalVAE(
        num_classes=len(moa_to_idx),
        img_channels=CONFIG["img_channels"],
        latent_dim=CONFIG["latent_dim"],
        cond_dim=CONFIG["cond_dim"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val = float("inf")
    history = []

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
        )

        for x, y in train_bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(x, y)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=CONFIG["beta"])
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += recon.item()
            train_kl_sum += kl.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
            )

        train_loss = train_loss_sum / len(train_loader)
        train_recon = train_recon_sum / len(train_loader)
        train_kl = train_kl_sum / len(train_loader)

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:03d} [val]",
            leave=False,
        )

        with torch.no_grad():
            for x, y in val_bar:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                x_hat, mu, logvar = model(x, y)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=CONFIG["beta"])

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

        epoch_time = time.time() - epoch_start

        log_msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val_loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} | "
            f"time={epoch_time:.1f}s"
        )
        logger.info(log_msg)

        history.append({
            "epoch": epoch,
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

        save_reconstructions(
            model, val_loader, epoch, run_dir, DEVICE,
            n_vis=CONFIG["n_vis_recon"]
        )

        save_class_samples_readable(
            model, moa_to_idx, epoch, run_dir, DEVICE,
            n_per_class=CONFIG["n_vis_per_class"]
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "moa_to_idx": moa_to_idx,
            "config": CONFIG,
            "val_loss": val_loss,
        }

        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            logger.info(f"Saved best model at epoch {epoch}")

    logger.info("Training complete.")

    # final export with best model
    best_ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    save_fixed_class_panel(model, moa_to_idx, run_dir, DEVICE, ckpt_name="best")


if __name__ == "__main__":
    main()
