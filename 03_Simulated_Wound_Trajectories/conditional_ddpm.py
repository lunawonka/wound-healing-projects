"""
bbbc021_ddpm.py — All-in-one Conditional DDPM for BBBC021 MCF7 cells
=====================================================================
Dataset  : BBBC021 (Week1_22123, 5 MoA classes)
Model    : Conditional DDPM with U-Net backbone + CFG
Channels : DAPI / Tubulin / Actin  →  (3, 64, 64)

Usage:
    python bbbc021_ddpm.py            # train from scratch
    python bbbc021_ddpm.py --test     # smoke-test only (no training)

Outputs:
    /data/olgameneg/checkpoints/      checkpoints
    /data/olgameneg/samples/          generated image grids every N epochs

Dependencies:
    pip install torch torchvision einops tifffile pandas
"""

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit only this section
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # ── Paths ─────────────────────────────────────────────────────────────────
    index_csv   = "/data/olgameneg/labeled_index_local.csv",
    base_dir    = "/data/olgameneg/images/",
    ckpt_dir    = "/data/olgameneg/checkpoints/",
    sample_dir  = "/data/olgameneg/samples/",

    # ── Data ──────────────────────────────────────────────────────────────────
    img_size    = 64,
    augment     = True,
    val_frac    = 0.10,       # fraction held out for validation

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = 5,          # MoA classes in Week1_22123
    base_ch     = 64,
    emb_dim     = 256,
    T           = 1000,       # diffusion timesteps

    # ── Training ──────────────────────────────────────────────────────────────
    epochs      = 200,
    batch_size  = 8,          # raise to 16 if GPU memory allows
    lr          = 1e-4,
    cfg_drop    = 0.10,       # CFG label-dropout probability
    num_workers = 4,

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every    = 10,        # print loss every N steps
    save_every   = 20,        # checkpoint every N epochs
    sample_every = 20,        # generate sample grid every N epochs
    guidance     = 7.5,       # CFG guidance scale at inference
)

# Null-class token index (reserved for CFG unconditional pass)
NULL_CLASS_IDX = CFG["num_classes"]   # = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def build_labeled_index(
    image_csv:  str,
    moa_csv:    str,
    output_csv: str,
) -> pd.DataFrame:
    """Join image metadata with MoA labels and fix path prefixes."""
    images  = pd.read_csv(image_csv)
    moa     = pd.read_csv(moa_csv)
    labeled = images.merge(
        moa,
        left_on  = ["Image_Metadata_Compound", "Image_Metadata_Concentration"],
        right_on = ["compound", "concentration"],
        how      = "inner",
    )
    for col in ["Image_PathName_DAPI", "Image_PathName_Tubulin", "Image_PathName_Actin"]:
        if col in labeled.columns:
            labeled[col] = labeled[col].str.split("/").str[-1]
    labeled = labeled.reset_index(drop=True)
    labeled.to_csv(output_csv, index=False)
    print(f"[data] {len(labeled)} labeled rows → {output_csv}")
    print(labeled["moa"].value_counts().to_string())
    return labeled


class BBBC021Dataset(Dataset):
    """
    3-channel fluorescence dataset for BBBC021.

    Loads DAPI / Tubulin / Actin TIFFs → fuses into (3, img_size, img_size)
    tensor normalised to [-1, 1].

    Label: MoA class index (int).
    """

    def __init__(
        self,
        df:       pd.DataFrame,
        base_dir: str,
        img_size: int  = 64,
        augment:  bool = False,
        moa2idx:  Optional[dict] = None,
    ):
        self.df       = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.img_size = img_size
        self.augment  = augment

        if moa2idx is None:
            classes      = sorted(self.df["moa"].unique())
            self.moa2idx = {m: i for i, m in enumerate(classes)}
        else:
            self.moa2idx = moa2idx

        self.idx2moa = {v: k for k, v in self.moa2idx.items()}
        self._resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _load_channel(self, path_col: str, fname_col: str, row: pd.Series) -> np.ndarray:
        fpath = self.base_dir / str(row[path_col]) / str(row[fname_col])
        if not fpath.exists():
            raise FileNotFoundError(
                f"TIFF not found: {fpath}\n"
                f"  base_dir={self.base_dir}  path={row[path_col]}  file={row[fname_col]}"
            )
        img = tifffile.imread(str(fpath)).astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > 0.5:
            img = TF.hflip(img)
        if torch.rand(1) > 0.5:
            img = TF.vflip(img)
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(img, k=k, dims=[-2, -1])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        dapi    = self._load_channel("Image_PathName_DAPI",    "Image_FileName_DAPI",    row)
        tubulin = self._load_channel("Image_PathName_Tubulin", "Image_FileName_Tubulin", row)
        actin   = self._load_channel("Image_PathName_Actin",   "Image_FileName_Actin",   row)

        img = torch.from_numpy(np.stack([dapi, tubulin, actin], axis=0))
        img = self._resize(img)
        if self.augment:
            img = self._augment(img)
        img = img * 2.0 - 1.0          # [0,1] → [-1,1]
        return img, self.moa2idx[row["moa"]]

    @property
    def num_classes(self) -> int:
        return len(self.moa2idx)

    def class_name(self, idx: int) -> str:
        return self.idx2moa.get(idx, "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1     = nn.GroupNorm(groups, in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(groups, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(self.act(cond)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = rearrange(qkv, "b (t c) h w -> t b (h w) c", t=3).unbind(0)
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1)
        out  = rearrange(attn @ v, "b (h w) c -> b c h w", h=H, w=W)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    """
    U-Net denoising backbone.
    Encoder : 64→128→256→512  (downsampling)
    Decoder : 512→256→128→64  (upsampling + skip connections)
    Conditioning: time + MoA class → scale+shift in every ResBlock
    """

    def __init__(
        self,
        img_channels: int   = 3,
        num_classes:  int   = 5,
        base_ch:      int   = 64,
        ch_mult:      tuple = (1, 2, 4, 8),
        emb_dim:      int   = 256,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.null_class_idx = num_classes   # CFG null token

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.cond_dim  = emb_dim

        chs = [base_ch * m for m in ch_mult]

        # Encoder
        self.input_conv  = nn.Conv2d(img_channels, chs[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = chs[0]
        self._enc_chs = [in_ch]

        for i, out_ch in enumerate(chs):
            use_attn = (i >= len(chs) - 2)
            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_ch, out_ch, self.cond_dim),
                ResBlock(out_ch, out_ch, self.cond_dim),
                SelfAttention(out_ch) if use_attn else nn.Identity(),
            ]))
            self._enc_chs.append(out_ch)
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock(chs[-1], chs[-1], self.cond_dim)
        self.mid_attn   = SelfAttention(chs[-1])
        self.mid_block2 = ResBlock(chs[-1], chs[-1], self.cond_dim)

        # Decoder
        self.up_blocks  = nn.ModuleList()
        self.upsamples  = nn.ModuleList()
        rev_chs = list(reversed(chs))

        for i, out_ch in enumerate(rev_chs):
            skip_ch  = self._enc_chs[-(i + 1)]
            use_attn = (i < 2)
            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_ch + skip_ch, out_ch, self.cond_dim),
                ResBlock(out_ch, out_ch, self.cond_dim),
                SelfAttention(out_ch) if use_attn else nn.Identity(),
            ]))
            if i < len(rev_chs) - 1:
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                ))
            in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], img_channels, 1)

    def forward(self, x, t, class_labels):
        cond = self.time_emb(t) + self.class_emb(class_labels)

        h = self.input_conv(x)
        skips = [h]
        for i, (rb1, rb2, attn) in enumerate(self.down_blocks):
            h = rb1(h, cond); h = rb2(h, cond); h = attn(h)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for i, (rb1, rb2, attn) in enumerate(self.up_blocks):
            h = torch.cat([h, skips[-(i + 1)]], dim=1)
            h = rb1(h, cond); h = rb2(h, cond); h = attn(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))


class DDPM(nn.Module):
    """
    DDPM wrapper: linear β schedule + forward diffusion + CFG sampling.
    """

    def __init__(self, unet: ConditionalUNet, T: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.unet = unet
        self.T    = T

        betas     = torch.linspace(beta_start, beta_end, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        abp       = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        pvar      = betas * (1 - abp) / (1 - alpha_bar)

        self.register_buffer("betas",             betas)
        self.register_buffer("sqrt_abar",         alpha_bar.sqrt())
        self.register_buffer("sqrt_1m_abar",      (1 - alpha_bar).sqrt())
        self.register_buffer("posterior_var",      pvar)
        self.register_buffer("posterior_mean_c1", betas * abp.sqrt() / (1 - alpha_bar))
        self.register_buffer("posterior_mean_c2", (1 - abp) * alphas.sqrt() / (1 - alpha_bar))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_abar[t][:, None, None, None]
        s2 = self.sqrt_1m_abar[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise

    def loss(self, x0, labels, cfg_drop_prob=0.10):
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        if cfg_drop_prob > 0:
            drop = torch.rand(B, device=x0.device) < cfg_drop_prob
            labels = labels.clone()
            labels[drop] = self.unet.null_class_idx
        return F.mse_loss(self.unet(x_t, t, labels), noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, labels, guidance_scale=1.0):
        tv = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        if guidance_scale > 1.0:
            ec = self.unet(x_t, tv, labels)
            eu = self.unet(x_t, tv, torch.full_like(labels, self.unet.null_class_idx))
            eps = eu + guidance_scale * (ec - eu)
        else:
            eps = self.unet(x_t, tv, labels)
        c1   = self.posterior_mean_c1[t]
        c2   = self.posterior_mean_c2[t]
        mean = c1 * x_t + c2 * (x_t - self.sqrt_1m_abar[t] * eps) / self.sqrt_abar[t]
        if t == 0:
            return mean
        return mean + self.posterior_var[t].sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, moa_class, n_samples=4, img_size=64,
               guidance_scale=7.5, device="cuda", verbose=True):
        self.eval()
        labels = torch.full((n_samples,), moa_class, dtype=torch.long, device=device)
        x = torch.randn(n_samples, 3, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            if verbose and t % 200 == 0:
                print(f"  sampling t={t:4d} …", end="\r")
            x = self.p_sample(x, t, labels, guidance_scale)
        if verbose:
            print()
        return (x.clamp(-1, 1) + 1) / 2


def build_model(cfg: dict) -> DDPM:
    ch_mult = (1, 2, 4, 8, 8) if cfg["img_size"] >= 128 else (1, 2, 4, 8)
    unet    = ConditionalUNet(
        img_channels=3,
        num_classes=cfg["num_classes"],
        base_ch=cfg["base_ch"],
        ch_mult=ch_mult,
        emb_dim=cfg["emb_dim"],
    )
    return DDPM(unet, T=cfg["T"])


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg):
    path = Path(cfg["ckpt_dir"]) / f"ddpm_epoch{epoch:04d}.pt"
    torch.save({
        "epoch": epoch, "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  [ckpt] → {path}")


def load_checkpoint(model, optimizer, scheduler, cfg, device):
    ckpts = sorted(Path(cfg["ckpt_dir"]).glob("ddpm_epoch*.pt"))
    if not ckpts:
        return 0
    state = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  [ckpt] resumed from {ckpts[-1]}  (epoch {state['epoch']})")
    return state["epoch"] + 1


@torch.no_grad()
def generate_samples(model, cfg, epoch, device, idx2moa):
    model.eval()
    imgs = []
    for cls in range(cfg["num_classes"]):
        imgs.append(model.sample(cls, n_samples=1, img_size=cfg["img_size"],
                                 guidance_scale=cfg["guidance"],
                                 device=device, verbose=False))
    grid = torch.cat(imgs, dim=0)
    path = Path(cfg["sample_dir"]) / f"samples_epoch{epoch:04d}.png"
    save_image(grid, path, nrow=cfg["num_classes"])
    labels = [idx2moa.get(i, str(i)) for i in range(cfg["num_classes"])]
    print(f"  [sample] → {path}")
    print(f"            {labels}")
    model.train()


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["sample_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BBBC021 Conditional DDPM")
    print(f"  device={device}  img={cfg['img_size']}px  "
          f"classes={cfg['num_classes']}  epochs={cfg['epochs']}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg["index_csv"])
    print(f"[data] {len(df)} labeled images")
    print(df["moa"].value_counts().to_string(), "\n")

    full_ds = BBBC021Dataset(df, cfg["base_dir"], cfg["img_size"], augment=cfg["augment"])
    idx2moa = full_ds.idx2moa

    n_val   = max(1, int(len(full_ds) * cfg["val_frac"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    print(f"[data] train={n_train}  val={n_val}  steps/epoch={len(train_loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {n_params:.1f} M parameters\n")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    start     = load_checkpoint(model, optimizer, scheduler, cfg, device)

    best_val  = float("inf")

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            loss = model.loss(imgs, labels, cfg_drop_prob=cfg["cfg_drop"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if (step + 1) % cfg["log_every"] == 0:
                avg = epoch_loss / (step + 1)
                print(f"  ep {epoch+1:4d} | step {step+1:3d}/{len(train_loader)} "
                      f"| loss {loss.item():.4f} | avg {avg:.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss += model.loss(imgs, labels, cfg_drop_prob=0.0).item()
        val_loss /= len(val_loader)

        avg_train = epoch_loss / len(train_loader)
        print(f"\nepoch {epoch+1:4d}/{cfg['epochs']} | "
              f"train {avg_train:.4f} | val {val_loss:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s\n")

        # Best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(cfg["ckpt_dir"]) / "ddpm_best.pt")
            print(f"  [best] val={val_loss:.4f} → ddpm_best.pt")

        if (epoch + 1) % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_train, cfg)

        if (epoch + 1) % cfg["sample_every"] == 0:
            generate_samples(model, cfg, epoch + 1, device, idx2moa)

    save_checkpoint(model, optimizer, scheduler, cfg["epochs"]-1, avg_train, cfg)
    print(f"\nDone. Best val loss: {best_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 — SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[smoke test]  device={device}")

    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters   : {n_params:.1f} M")
    print(f"Num classes  : {cfg['num_classes']}  (null idx={NULL_CLASS_IDX})")

    B      = 4
    x0     = torch.randn(B, 3, cfg["img_size"], cfg["img_size"], device=device)
    labels = torch.randint(0, cfg["num_classes"], (B,), device=device)
    loss   = model.loss(x0, labels)
    print(f"Loss         : {loss.item():.4f}  (expect ~1.0)")

    t_vec  = torch.randint(0, cfg["T"], (B,), device=device)
    x_t, _ = model.q_sample(x0, t_vec)
    pred   = model.unet(x_t, t_vec, labels)
    print(f"U-Net output : {pred.shape}")

    model.T = 10
    samples = model.sample(0, n_samples=2, img_size=cfg["img_size"],
                           device=device, verbose=False)
    print(f"Sample shape : {samples.shape}")
    print(f"Value range  : [{samples.min():.3f}, {samples.max():.3f}]")
    print("\nAll checks passed ✓")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="smoke test only")
    args = parser.parse_args()

    if args.test:
        smoke_test(CFG)
    else:
        train(CFG)