"""
Step 5 — Latent Diffusion Model (LDM) using MONAI Generative
Two-phase training:
  Phase 1: AutoencoderKL  — compresses 512×512 → 64×64 latent space
  Phase 2: DiffusionModelUNet — learns to denoise in latent space
Conditioning: cell line label + wound area %
GPU: NVIDIA A100
Reference: Pinaya et al. 2023 (MONAI Generative — your proposal ref [5])
"""
import os, time, math, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import LatentDiffusionInferer

# ── CONFIG ────────────────────────────────────────────────
BASE_DIR      = Path("/data/bbbc_019_clean")
PROC_DIR      = BASE_DIR / "processed/images"
META_FILE     = BASE_DIR / "results/preprocessing/dataset_metadata.csv"
OUT_DIR       = BASE_DIR / "results/ldm"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE        = torch.device("cuda")
IMG_SIZE      = 512
LATENT_CH     = 3        # latent channels — compression factor 512→64
BATCH         = 4
AE_EPOCHS     = 100      # autoencoder training epochs
DM_EPOCHS     = 200      # diffusion model training epochs
AE_LR         = 1e-4
DM_LR         = 1e-4
T_STEPS       = 1000
N_CELL_LINES  = 5

print("="*60)
print("  LATENT DIFFUSION MODEL — MONAI GENERATIVE")
print("="*60)
print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
print(f"  Image size  : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Latent ch   : {LATENT_CH} (compression: 512→64)")
print(f"  AE epochs   : {AE_EPOCHS}")
print(f"  DM epochs   : {DM_EPOCHS}")
print("="*60 + "\n")


# ════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════
class WoundDataset(Dataset):
    def __init__(self, proc_dir, meta_file):
        self.proc_dir = Path(proc_dir)
        meta          = pd.read_csv(meta_file)

        self.samples  = []
        for _, row in meta.iterrows():
            path = self.proc_dir / f"{row.stem}.npy"
            if path.exists():
                self.samples.append({
                    "path":      str(path),
                    "cell_line": int(row.cell_line_id),
                    "wound_pct": float(row.wound_area_pct) / 100.0,
                })
        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = np.load(s["path"]).astype(np.float32)
        # Normalise to [-1, 1] for autoencoder
        img_t  = torch.tensor(img).unsqueeze(0) * 2 - 1
        cl_t   = torch.tensor(s["cell_line"], dtype=torch.long)
        wnd_t  = torch.tensor(s["wound_pct"], dtype=torch.float32)
        return img_t, cl_t, wnd_t


# ════════════════════════════════════════════════════════════
# PHASE 1 — AUTOENCODER KL
# ════════════════════════════════════════════════════════════
def build_autoencoder():
    """
    AutoencoderKL: compresses 512×512 → 64×64 latent space.
    KL regularisation keeps latent distribution close to Gaussian
    which is required for the diffusion model to work in latent space.
    8× spatial compression (512/8 = 64).
    """
    ae = AutoencoderKL(
        spatial_dims     = 2,
        in_channels      = 1,
        out_channels     = 1,
        num_channels     = (128, 256, 512),   # encoder channel widths
        latent_channels  = LATENT_CH,
        num_res_blocks   = 2,
        norm_num_groups  = 32,
        attention_levels = (False, False, True),
    ).to(DEVICE)
    n = sum(p.numel() for p in ae.parameters())
    print(f"AutoencoderKL parameters: {n:,}")
    return ae


def train_autoencoder(ds):
    print("\n" + "="*50)
    print("  PHASE 1 — TRAINING AUTOENCODER KL")
    print("="*50)

    n_train = int(0.9 * len(ds))
    n_val   = len(ds) - n_train
    tr_ds, val_ds = random_split(ds, [n_train, n_val])
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                        num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

    ae     = build_autoencoder()
    opt    = optim.Adam(ae.parameters(), lr=AE_LR)
    sched  = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=AE_EPOCHS, eta_min=1e-6)

    recon_loss_fn = nn.L1Loss()

    tr_losses, val_losses = [], []
    best_val  = float("inf")
    t_start   = time.time()

    for epoch in range(1, AE_EPOCHS + 1):
        # Train
        ae.train()
        ep_loss = 0.0
        for imgs, _, _ in tr_dl:
            imgs = imgs.to(DEVICE)
            recon, z_mu, z_sigma = ae(imgs)

            # Reconstruction loss
            recon_loss = recon_loss_fn(recon, imgs)

            # KL divergence loss — regularises latent space
            kl_loss    = 0.5 * torch.mean(
                z_mu.pow(2) + z_sigma.pow(2)
                - torch.log(z_sigma.pow(2)) - 1
            )
            loss = recon_loss + 1e-6 * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(tr_dl)
        tr_losses.append(ep_loss)
        sched.step()

        # Validate
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, _ in val_dl:
                imgs  = imgs.to(DEVICE)
                recon, z_mu, z_sigma = ae(imgs)
                val_loss += recon_loss_fn(recon, imgs).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        elapsed = (time.time() - t_start) / 3600
        print(f"AE Epoch {epoch:4d}/{AE_EPOCHS} | "
              f"Train: {ep_loss:.5f} | "
              f"Val: {val_loss:.5f} | "
              f"Time: {elapsed:.2f}h")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "loss":  best_val,
                "gpu":   torch.cuda.get_device_name(0),
                "model": ae.state_dict(),
            }, OUT_DIR / "best_autoencoder.pth")

        # Save last epoch always
        torch.save({
            "epoch": epoch,
            "loss":  val_loss,
            "gpu":   torch.cuda.get_device_name(0),
            "model": ae.state_dict(),
        }, OUT_DIR / "last_autoencoder.pth")

        # Save reconstruction visualisation every 25 epochs
        if epoch % 25 == 0 or epoch == AE_EPOCHS:
            ae.eval()
            with torch.no_grad():
                sample_imgs = next(iter(val_dl))[0][:4].to(DEVICE)
                recon, _, _ = ae(sample_imgs)

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for i in range(4):
                orig = (sample_imgs[i].squeeze().cpu().numpy() + 1) / 2
                rec  = (recon[i].squeeze().cpu().numpy() + 1) / 2
                axes[0][i].imshow(orig, cmap="gray")
                axes[0][i].set_title("Original", fontsize=8)
                axes[1][i].imshow(rec,  cmap="gray")
                axes[1][i].set_title("Reconstructed", fontsize=8)
                for ax in [axes[0][i], axes[1][i]]:
                    ax.axis("off")

            plt.suptitle(
                f"AutoencoderKL Reconstruction | Epoch {epoch}\n"
                f"NVIDIA {torch.cuda.get_device_name(0)} | "
                f"Val loss: {val_loss:.5f}",
                fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"ae_recon_epoch_{epoch:04d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

    # Plot AE loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(tr_losses,  label="Train", color="#76b900", lw=2)
    plt.plot(val_losses, label="Val",   color="#ff6600", lw=2)
    plt.title(f"AutoencoderKL Loss | NVIDIA A100\nBest val: {best_val:.5f}",
              fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("L1 Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ae_loss_curve.png", dpi=150)
    plt.close()

    print(f"\n✓ Autoencoder trained | Best val loss: {best_val:.5f}")
    return ae


# ════════════════════════════════════════════════════════════
# PHASE 2 — DIFFUSION MODEL IN LATENT SPACE
# ════════════════════════════════════════════════════════════

class ConditionEmbedder(nn.Module):
    """
    Embeds cell line + wound_pct into a cross-attention context
    tensor for the DiffusionModelUNet.
    Shape: (B, seq_len=1, context_dim=512)
    """
    def __init__(self, n_cell_lines=5, context_dim=512):
        super().__init__()
        self.cl_emb   = nn.Embedding(n_cell_lines, context_dim)
        self.wnd_proj = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.SiLU(),
        )

    def forward(self, cell_line, wound_pct):
        cl_e  = self.cl_emb(cell_line)
        wnd_e = self.wnd_proj(wound_pct.unsqueeze(1))
        fused = self.fuse(torch.cat([cl_e, wnd_e], dim=1))
        # Add sequence dimension for cross-attention
        return fused.unsqueeze(1)   # (B, 1, context_dim)


def build_diffusion_model():
    """
    DiffusionModelUNet operates in latent space (64×64×3).
    Uses cross-attention for conditioning on cell line + wound area.
    """
    unet = DiffusionModelUNet(
        spatial_dims          = 2,
        in_channels           = LATENT_CH,
        out_channels          = LATENT_CH,
        num_res_blocks        = 2,
        num_channels          = (256, 512, 768),
        attention_levels      = (False, True, True),
        num_head_channels     = (0, 512, 768),
        # Cross-attention for conditioning
        with_conditioning     = True,
        cross_attention_dim   = 512,
    ).to(DEVICE)
    n = sum(p.numel() for p in unet.parameters())
    print(f"DiffusionModelUNet parameters: {n:,}")
    return unet


def train_diffusion_model(ae, ds):
    print("\n" + "="*50)
    print("  PHASE 2 — TRAINING DIFFUSION MODEL IN LATENT SPACE")
    print("="*50)

    dl      = DataLoader(ds, batch_size=BATCH, shuffle=True,
                         num_workers=4, pin_memory=True)

    unet    = build_diffusion_model()
    embedder = ConditionEmbedder(
        n_cell_lines=N_CELL_LINES, context_dim=512
    ).to(DEVICE)

    # MONAI DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps = T_STEPS,
        schedule            = "scaled_linear_beta",
        beta_start          = 0.0015,
        beta_end            = 0.0195,
    )

    # MONAI LDM inferer — handles latent encoding/decoding
    # scaling_factor adapts latent std to be ~Gaussian
    with torch.no_grad():
        sample_batch = next(iter(dl))[0][:4].to(DEVICE)
        z, _, _      = ae(sample_batch)
        scale_factor = 1.0 / z.std()
        print(f"Latent scaling factor: {scale_factor:.4f}")

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    all_params = list(unet.parameters()) + list(embedder.parameters())
    opt        = optim.AdamW(all_params, lr=DM_LR, weight_decay=1e-5)
    sched_lr   = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=DM_EPOCHS, eta_min=1e-6)
    loss_fn    = nn.MSELoss()

    # Freeze autoencoder during diffusion training
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    losses    = []
    best_loss = float("inf")
    t_start   = time.time()

    print(f"Training diffusion model | {len(ds)} samples | "
          f"{len(dl)} batches/epoch\n")

    for epoch in range(1, DM_EPOCHS + 1):
        unet.train()
        embedder.train()
        ep_loss = 0.0

        for imgs, cell_line, wound_pct in dl:
            imgs      = imgs.to(DEVICE)
            cell_line = cell_line.to(DEVICE)
            wound_pct = wound_pct.to(DEVICE)

            # Build conditioning context
            context = embedder(cell_line, wound_pct)

            # Sample noise and timesteps
            noise = torch.randn_like(
                torch.zeros(imgs.shape[0], LATENT_CH,
                            IMG_SIZE // 8, IMG_SIZE // 8).to(DEVICE)
            )
            t = torch.randint(
                0, T_STEPS, (imgs.shape[0],), device=DEVICE).long()

            # MONAI inferer handles: encode → add noise → predict noise
            noise_pred = inferer(
                inputs           = imgs,
                autoencoder_model = ae,
                diffusion_model  = unet,
                noise            = noise,
                timesteps        = t,
                condition        = context,
            )

            loss = loss_fn(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(dl)
        losses.append(ep_loss)
        sched_lr.step()

        elapsed = (time.time() - t_start) / 3600
        mem     = torch.cuda.memory_allocated() / 1e9
        print(f"DM Epoch {epoch:4d}/{DM_EPOCHS} | "
              f"Loss: {ep_loss:.5f} | "
              f"GPU: {mem:.1f}GB | "
              f"Time: {elapsed:.2f}h")

        # Save best + last
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({
                "epoch":    epoch,
                "loss":     best_loss,
                "gpu":      torch.cuda.get_device_name(0),
                "unet":     unet.state_dict(),
                "embedder": embedder.state_dict(),
                "scale":    scale_factor,
            }, OUT_DIR / "best_ldm.pth")
            print(f"  ✓ New best {best_loss:.5f} — saved best_ldm.pth")

        torch.save({
            "epoch":    epoch,
            "loss":     ep_loss,
            "gpu":      torch.cuda.get_device_name(0),
            "unet":     unet.state_dict(),
            "embedder": embedder.state_dict(),
            "scale":    scale_factor,
        }, OUT_DIR / "last_ldm.pth")

        # Generate samples + loss curve every 25 epochs
        if epoch % 25 == 0 or epoch == DM_EPOCHS:
            generate_ldm_samples(ae, unet, embedder,
                                 scheduler, scale_factor, epoch)

            plt.figure(figsize=(10, 4))
            plt.plot(losses, color="#76b900", lw=2)
            plt.title(
                f"LDM Diffusion Loss | NVIDIA A100 | Epoch {epoch}\n"
                f"Best: {best_loss:.5f}",
                fontweight="bold")
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "ldm_loss_curve.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

    total_h = (time.time() - t_start) / 3600
    print(f"\n✓ Diffusion model trained | "
          f"Best loss: {best_loss:.5f} | "
          f"Time: {total_h:.2f}h")
    return unet, embedder, scheduler, scale_factor


# ════════════════════════════════════════════════════════════
# GENERATION
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_ldm_samples(ae, unet, embedder, scheduler,
                          scale_factor, epoch):
    """
    Generate synthetic wound images using DDIM scheduler
    (50 steps instead of 1000 — 20× faster sampling).
    """
    ae.eval(); unet.eval(); embedder.eval()

    # Use DDIM for fast sampling at inference
    ddim = DDIMScheduler(
        num_train_timesteps = T_STEPS,
        schedule            = "scaled_linear_beta",
        beta_start          = 0.0015,
        beta_end            = 0.0195,
        clip_sample         = False,
    )
    ddim.set_timesteps(num_inference_steps=50)

    inferer = LatentDiffusionInferer(ddim, scale_factor=scale_factor)

    conditions = [
        (0,  5.0, "DA3 — 5% wound"),
        (0, 30.0, "DA3 — 30% wound"),
        (0, 55.0, "DA3 — 55% wound"),
        (1, 25.0, "Melanoma — 25% wound"),
        (2, 15.0, "MDCK — 15% wound"),
        (3, 30.0, "HEK293T — 30% wound"),
    ]

    fig, axes = plt.subplots(len(conditions), 4,
                             figsize=(16, 4 * len(conditions)))

    for row, (cl_id, wnd_pct, label) in enumerate(conditions):
        n = 4
        cl  = torch.full((n,), cl_id,
                          dtype=torch.long).to(DEVICE)
        wnd = torch.full((n,), wnd_pct / 100.0).to(DEVICE)
        ctx = embedder(cl, wnd)

        # Start from random noise in latent space
        latent_shape = (n, LATENT_CH,
                        IMG_SIZE // 8, IMG_SIZE // 8)
        noise        = torch.randn(latent_shape).to(DEVICE)

        # MONAI inferer handles reverse diffusion + decoding
        imgs = inferer.sample(
            input_noise       = noise,
            autoencoder_model = ae,
            diffusion_model   = unet,
            scheduler         = ddim,
            conditioning      = ctx,
        )

        for col in range(4):
            img_np = imgs[col].squeeze().cpu().numpy()
            img_np = (img_np - img_np.min()) / (
                img_np.max() - img_np.min() + 1e-8)
            axes[row][col].imshow(img_np, cmap="gray")
            axes[row][col].set_title(label if col == 0 else "",
                                     fontsize=8)
            axes[row][col].axis("off")

    plt.suptitle(
        f"LDM Synthetic Wound Images | Epoch {epoch}\n"
        f"NVIDIA {torch.cuda.get_device_name(0)} | "
        f"MONAI Generative | DDIM 50 steps\n"
        f"Rows 1-3: DA3 wound 5%→30%→55% (conditioning check)\n"
        f"Proposal Month 3-4",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    path = OUT_DIR / f"ldm_samples_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved LDM samples → {path.name}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    ds = WoundDataset(PROC_DIR, META_FILE)

    # ── Phase 1: Autoencoder ──────────────────────────────
    ae_ckpt = OUT_DIR / "last_autoencoder.pth"
    if ae_ckpt.exists():
        print(f"\nLoading existing autoencoder: {ae_ckpt}")
        ae = build_autoencoder()
        ck = torch.load(str(ae_ckpt), map_location=DEVICE)
        ae.load_state_dict(ck["model"])
        print(f"  Loaded from epoch {ck['epoch']}")
    else:
        ae = train_autoencoder(ds)

    # ── Phase 2: Diffusion model ──────────────────────────
    ldm_ckpt = OUT_DIR / "last_ldm.pth"
    if ldm_ckpt.exists():
        print(f"\nLoading existing LDM: {ldm_ckpt}")
        unet     = build_diffusion_model()
        embedder = ConditionEmbedder(N_CELL_LINES, 512).to(DEVICE)
        ck       = torch.load(str(ldm_ckpt), map_location=DEVICE)
        unet.load_state_dict(ck["unet"])
        embedder.load_state_dict(ck["embedder"])
        scale_factor = ck["scale"]

        scheduler = DDPMScheduler(
            num_train_timesteps = T_STEPS,
            schedule            = "scaled_linear_beta",
            beta_start          = 0.0015,
            beta_end            = 0.0195,
        )
        print(f"  Loaded from epoch {ck['epoch']}")
        print("  Generating samples from loaded checkpoint...")
        generate_ldm_samples(ae, unet, embedder,
                             scheduler, scale_factor,
                             epoch=ck["epoch"])
    else:
        unet, embedder, scheduler, scale_factor = \
            train_diffusion_model(ae, ds)

    print(f"\n{'='*60}")
    print(f"  LDM COMPLETE")
    print(f"{'='*60}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Autoencoder  : {OUT_DIR}/best_autoencoder.pth")
    print(f"  Diffusion    : {OUT_DIR}/best_ldm.pth")
    print(f"  Samples      : {OUT_DIR}/ldm_samples_*.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()