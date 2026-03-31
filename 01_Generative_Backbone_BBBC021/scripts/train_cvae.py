import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image

from dataset_bbbc021_patches import BBBC021PatchDataset
from model.model_cvae import ConditionalVAE


ROOT = Path("/data/annapan/prop/bbbc021")
PATCH_METADATA = ROOT / "patches_256_metadata.csv"
MOA_JSON = ROOT / "moa_to_idx.json"
OUT_DIR = ROOT / "runs" / "cvae_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 8
LATENT_DIM = 128
COND_DIM = 32
LR = 1e-4
EPOCHS = 30
BETA = 0.001


def denorm(x):
    return (x + 1.0) / 2.0


def make_weighted_sampler(dataset):
    labels = [dataset.moa_to_idx[m] for m in dataset.df["moa"].tolist()]
    class_counts = torch.bincount(torch.tensor(labels), minlength=len(dataset.moa_to_idx)).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def vae_loss(x_hat, x, mu, logvar, beta=0.001):
    recon = F.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon, kl


def save_reconstructions(model, loader, epoch, device):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        x_hat, _, _ = model(x, y)

    grid = torch.cat([x[:8], x_hat[:8]], dim=0)
    save_image(
        denorm(grid).clamp(0, 1),
        OUT_DIR / f"recon_epoch_{epoch:03d}.png",
        nrow=8
    )


def save_class_samples(model, moa_to_idx, epoch, device, n_per_class=8):
    model.eval()
    idx_to_moa = {v: k for k, v in moa_to_idx.items()}

    all_imgs = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(len(moa_to_idx)):
            y = torch.full((n_per_class,), class_idx, dtype=torch.long, device=device)
            x_gen = model.sample(y, device=device)
            all_imgs.append(x_gen.cpu())
            all_labels.extend([idx_to_moa[class_idx]] * n_per_class)

    all_imgs = torch.cat(all_imgs, dim=0)

    save_image(
        denorm(all_imgs).clamp(0, 1),
        OUT_DIR / f"samples_epoch_{epoch:03d}.png",
        nrow=n_per_class
    )


def main():
    with open(MOA_JSON, "r") as f:
        moa_to_idx = json.load(f)

    train_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="train",
        moa_to_idx=moa_to_idx,
    )

    val_dataset = BBBC021PatchDataset(
        metadata_csv=PATCH_METADATA,
        split="val",
        moa_to_idx=moa_to_idx,
    )

    train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ConditionalVAE(
        num_classes=len(moa_to_idx),
        img_channels=3,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, y)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                x_hat, mu, logvar = model(x, y)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=BETA)

                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()

        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val_loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f}"
        )

        save_reconstructions(model, val_loader, epoch, DEVICE)
        save_class_samples(model, moa_to_idx, epoch, DEVICE, n_per_class=8)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "moa_to_idx": moa_to_idx,
            "val_loss": val_loss,
        }

        torch.save(ckpt, OUT_DIR / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, OUT_DIR / "best.pt")
            print(f"Saved best model at epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()
