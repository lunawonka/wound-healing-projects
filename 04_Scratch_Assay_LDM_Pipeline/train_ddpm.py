"""
Step 4 — Conditional DDPM Training
Dataset: BBBC019 (660 preprocessed samples → downsampled to 256x256)
Conditioning: cell line label + wound area %
Proposal: Month 3-4 — Generative model training
GPU: NVIDIA A100
"""
import os, time, math, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────
BASE_DIR      = Path("/data/bbbc_019_clean")
PROC_DIR      = BASE_DIR / "processed/images"
PROC_MASK_DIR = BASE_DIR / "processed/masks"
META_FILE     = BASE_DIR / "results/preprocessing/dataset_metadata.csv"
OUT_DIR       = BASE_DIR / "results/ddpm"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE      = 256        # downsample from 512 → 256 for DDPM
BATCH         = 8          # safe for A100 at 256x256
EPOCHS        = 200
LR            = 2e-4
T_STEPS       = 1000       # diffusion timesteps
N_CELL_LINES  = 5          # DA3, Melanoma, MDCK, HEK293T, Unknown

print("="*60)
print("  STEP 4 — CONDITIONAL DDPM TRAINING")
print("="*60)
print(f"  Timestamp  : {datetime.datetime.now()}")
print(f"  GPU        : {torch.cuda.get_device_name(0)}")
print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print(f"  Image size : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH}")
print(f"  Timesteps  : {T_STEPS}")
print("="*60 + "\n")


# ════════════════════════════════════════════════════════════
# 1. DATASET
# ════════════════════════════════════════════════════════════
class WoundDataset(Dataset):
    """
    Loads preprocessed 512x512 .npy wound images,
    downsamples to 256x256 for DDPM training.
    Returns: image tensor, cell_line label, wound_area_pct
    """
    def __init__(self, proc_dir, meta_file, img_size=256):
        self.proc_dir = Path(proc_dir)
        self.img_size = img_size
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

        print(f"Dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = np.load(s["path"]).astype(np.float32)   # 512×512

        # Downsample 512 → 256 using PIL
        pil  = Image.fromarray((img * 255).astype(np.uint8))
        pil  = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        img  = np.array(pil).astype(np.float32) / 255.0

        # Normalise to [-1, 1] — standard for diffusion models
        img_t  = torch.tensor(img).unsqueeze(0) * 2 - 1

        cl_t   = torch.tensor(s["cell_line"], dtype=torch.long)
        wnd_t  = torch.tensor(s["wound_pct"], dtype=torch.float32)

        return img_t, cl_t, wnd_t


# ════════════════════════════════════════════════════════════
# 2. DIFFUSION NOISE SCHEDULE
# ════════════════════════════════════════════════════════════
# Linear beta schedule — standard from Ho et al. 2020 (DDPM paper)
betas      = torch.linspace(1e-4, 0.02, T_STEPS).to(DEVICE)
alphas     = 1.0 - betas
alpha_bar  = torch.cumprod(alphas, dim=0)   # cumulative product

def q_sample(x0, t, noise=None):
    """
    Forward diffusion: add noise to image x0 at timestep t.
    Returns noisy image xt and the noise that was added.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    ab_t  = alpha_bar[t].view(-1, 1, 1, 1)
    xt    = torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * noise
    return xt, noise


# ════════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURE — Conditional UNet Denoiser
# ════════════════════════════════════════════════════════════

class SinusoidalPositionEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t as a dense vector.
    Standard in all DDPM implementations.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs    = torch.exp(
            -math.log(10000) *
            torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """
    Core building block of the UNet.
    Injects both timestep embedding and conditioning vector
    via learned linear projections.
    """
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act      = nn.SiLU()

        # Project timestep + condition into channel space
        self.emb_proj = nn.Linear(emb_dim, out_ch)

        # Skip connection when channels change
        self.skip     = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x, emb):
        h  = self.act(self.norm1(x))
        h  = self.conv1(h)
        # Add conditioning — broadcast over spatial dims
        h  = h + self.emb_proj(emb)[:, :, None, None]
        h  = self.act(self.norm2(h))
        h  = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention at the bottleneck.
    Helps model capture long-range spatial dependencies
    — important for wound structure which spans the image.
    """
    def __init__(self, ch):
        super().__init__()
        self.norm  = nn.GroupNorm(8, ch)
        self.attn  = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h  = self.norm(x)
        # Reshape spatial dims into sequence for attention
        h  = h.view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h  = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class ConditionalUNet(nn.Module):
    """
    Conditional UNet denoiser for DDPM.

    Conditioning inputs:
      - t          : diffusion timestep (sinusoidal embedding)
      - cell_line  : integer label → learned embedding
      - wound_pct  : scalar 0-1 → linear projection

    All three are fused into a single conditioning vector
    that modulates every ResidualBlock via feature-wise addition.

    Architecture:
      Encoder: 3 downsampling stages (256→128→64→32)
      Bottleneck: 2 ResBlocks + Self-Attention
      Decoder: 3 upsampling stages with skip connections
    """
    def __init__(self, base_ch=64, emb_dim=256,
                 n_cell_lines=5):
        super().__init__()

        # ── Conditioning encoders ────────────────────────
        self.t_enc  = nn.Sequential(
            SinusoidalPositionEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cl_emb  = nn.Embedding(n_cell_lines, emb_dim)
        self.wnd_proj = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Fuse all three conditioning signals
        self.cond_fuse = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.SiLU(),
        )

        # ── Encoder ──────────────────────────────────────
        # Input: (B, 1, 256, 256)
        self.enc_in   = nn.Conv2d(1, base_ch, 3, padding=1)

        self.enc1     = ResidualBlock(base_ch,   base_ch*2, emb_dim)
        self.down1    = nn.Conv2d(base_ch*2, base_ch*2, 4, stride=2, padding=1)

        self.enc2     = ResidualBlock(base_ch*2, base_ch*4, emb_dim)
        self.down2    = nn.Conv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)

        self.enc3     = ResidualBlock(base_ch*4, base_ch*8, emb_dim)
        self.down3    = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)

        # ── Bottleneck ────────────────────────────────────
        self.bot1     = ResidualBlock(base_ch*8, base_ch*8, emb_dim)
        self.bot_attn = AttentionBlock(base_ch*8)
        self.bot2     = ResidualBlock(base_ch*8, base_ch*8, emb_dim)

        # ── Decoder ───────────────────────────────────────
        self.up3      = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        self.dec3     = ResidualBlock(base_ch*16, base_ch*4, emb_dim)  # *16 = skip concat

        self.up2      = nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)
        self.dec2     = ResidualBlock(base_ch*8,  base_ch*2, emb_dim)

        self.up1      = nn.ConvTranspose2d(base_ch*2, base_ch*2, 4, stride=2, padding=1)
        self.dec1     = ResidualBlock(base_ch*4,  base_ch,   emb_dim)

        # ── Output head ───────────────────────────────────
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ConditionalUNet parameters: {n_params:,}")

    def forward(self, x, t, cell_line, wound_pct):
        # Build conditioning vector
        t_e   = self.t_enc(t)
        cl_e  = self.cl_emb(cell_line)
        wnd_e = self.wnd_proj(wound_pct.unsqueeze(1))
        cond  = self.cond_fuse(torch.cat([t_e, cl_e, wnd_e], dim=1))

        # Encoder
        h0    = self.enc_in(x)
        h1    = self.enc1(h0,            cond)
        h2    = self.enc2(self.down1(h1), cond)
        h3    = self.enc3(self.down2(h2), cond)

        # Bottleneck
        hb    = self.bot1(self.down3(h3), cond)
        hb    = self.bot_attn(hb)
        hb    = self.bot2(hb,             cond)

        # Decoder — skip connections concatenated
        h     = self.dec3(
            torch.cat([self.up3(hb), h3], dim=1), cond)
        h     = self.dec2(
            torch.cat([self.up2(h),  h2], dim=1), cond)
        h     = self.dec1(
            torch.cat([self.up1(h),  h1], dim=1), cond)

        return self.out_conv(self.out_act(self.out_norm(h)))


# ════════════════════════════════════════════════════════════
# 4. SAMPLING (DDPM REVERSE PROCESS)
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def ddpm_sample(model, cell_line_id, wound_pct_val,
                n=4, img_size=256):
    """
    Generate n synthetic wound images conditioned on
    cell_line_id and wound_pct_val (0-100).
    Uses the full DDPM reverse process (T=1000 steps).
    """
    model.eval()
    x      = torch.randn(n, 1, img_size, img_size).to(DEVICE)
    cl     = torch.full((n,), cell_line_id,
                        dtype=torch.long).to(DEVICE)
    wnd    = torch.full((n,), wound_pct_val / 100.0).to(DEVICE)

    for t_val in reversed(range(T_STEPS)):
        t      = torch.full((n,), t_val,
                            dtype=torch.long).to(DEVICE)
        eps    = model(x, t, cl, wnd)
        b_t    = betas[t_val]
        a_t    = alphas[t_val]
        ab_t   = alpha_bar[t_val]

        # Reverse diffusion step
        x      = (1 / torch.sqrt(a_t)) * (
            x - (b_t / torch.sqrt(1 - ab_t)) * eps
        )
        if t_val > 0:
            x += torch.sqrt(b_t) * torch.randn_like(x)

    # Rescale from [-1,1] to [0,1]
    return (x.clamp(-1, 1) + 1) / 2


def save_generated_samples(model, epoch):
    """
    Generate synthetic wound images for all conditions
    and save a grid. This is Step 5 (generation) happening
    at checkpoints during training.
    """
    conditions = [
        (0, 10.0, "DA3 — small wound (10%)"),
        (0, 40.0, "DA3 — large wound (40%)"),
        (1, 25.0, "Melanoma — 25% wound"),
        (2, 15.0, "MDCK — 15% wound"),
        (3, 30.0, "HEK293T — 30% wound"),
    ]

    fig, axes = plt.subplots(len(conditions), 4,
                             figsize=(16, 4 * len(conditions)))

    for row, (cl_id, wnd_pct, label) in enumerate(conditions):
        imgs = ddpm_sample(model, cl_id, wnd_pct, n=4)
        for col in range(4):
            axes[row][col].imshow(
                imgs[col].squeeze().cpu().numpy(), cmap="gray")
            axes[row][col].set_title(
                label if col == 0 else "", fontsize=8)
            axes[row][col].axis("off")

    plt.suptitle(
        f"Conditional DDPM — Synthetic Wound Images | Epoch {epoch}\n"
        f"NVIDIA {torch.cuda.get_device_name(0)} | "
        f"Conditioning: cell line + wound area %\n"
        f"Proposal Month 3-4",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = OUT_DIR / f"generated_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved generated samples → {path.name}")


# ════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ════════════════════════════════════════════════════════════
def train():
    # Dataset + dataloader
    ds      = WoundDataset(PROC_DIR, META_FILE, img_size=IMG_SIZE)
    dl      = DataLoader(ds, batch_size=BATCH, shuffle=True,
                         num_workers=4, pin_memory=True)

    # Model
    model   = ConditionalUNet(
        base_ch=64, emb_dim=256,
        n_cell_lines=N_CELL_LINES
    ).to(DEVICE)

    # Optimiser + scheduler
    opt     = optim.AdamW(model.parameters(),
                          lr=LR, weight_decay=1e-5)
    sched   = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    losses   = []
    best_loss = float("inf")
    t_start   = time.time()

    print(f"Starting training — {len(ds)} samples, "
          f"{len(dl)} batches/epoch\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0

        for x0, cell_line, wound_pct in dl:
            x0        = x0.to(DEVICE)
            cell_line = cell_line.to(DEVICE)
            wound_pct = wound_pct.to(DEVICE)

            # Sample random timestep for each image in batch
            t         = torch.randint(
                0, T_STEPS, (x0.size(0),), device=DEVICE)

            # Forward diffusion — add noise
            xt, noise = q_sample(x0, t)

            # Predict the noise that was added
            pred_noise = model(xt, t, cell_line, wound_pct)

            # Loss = MSE between predicted and actual noise
            loss       = loss_fn(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item()

        ep_loss /= len(dl)
        losses.append(ep_loss)
        sched.step()

        elapsed_h = (time.time() - t_start) / 3600
        mem_gb    = torch.cuda.memory_allocated() / 1e9

        print(f"Epoch {epoch:4d}/{EPOCHS} | "
              f"Loss: {ep_loss:.5f} | "
              f"GPU: {mem_gb:.1f}GB | "
              f"Time: {elapsed_h:.2f}h")

        # Save best model checkpoint
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({
                "epoch":         epoch,
                "loss":          best_loss,
                "gpu":           torch.cuda.get_device_name(0),
                "model":         model.state_dict(),
                "n_cell_lines":  N_CELL_LINES,
                "img_size":      IMG_SIZE,
                "timestamp":     str(datetime.datetime.now()),
            }, OUT_DIR / "best_model_a100.pth")

        # Generate samples every 25 epochs + final epoch
        if epoch % 25 == 0 or epoch == EPOCHS:
            save_generated_samples(model, epoch)

            # Save loss curve so far
            plt.figure(figsize=(10, 4))
            plt.plot(losses, color="#76b900", lw=2)
            plt.title(
                f"DDPM Training Loss | NVIDIA A100 | Epoch {epoch}\n"
                f"Best: {best_loss:.5f}",
                fontweight="bold")
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "loss_curve.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

    # ── Final summary ─────────────────────────────────────
    total_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  STEP 4 COMPLETE — DDPM TRAINING")
    print(f"{'='*60}")
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Best loss  : {best_loss:.5f}")
    print(f"  Total time : {total_h:.2f} hours")
    print(f"  Checkpoint : {OUT_DIR}/best_model_a100.pth")
    print(f"{'='*60}")

    return model, losses


if __name__ == "__main__":
    model, losses = train()