# Conditional Diffusion Models for Wound-Healing Microscopy (BBBC019)

This repository contains a full pipeline for **preprocessing**, **visualising**, and **training conditional generative models** on the **BBBC019 wound-healing assay dataset**. The code is designed around a research workflow that starts from paired microscopy images and wound masks, builds a curated training set, and then trains two image-generation models:

* a **conditional DDPM** operating directly in image space at **256×256**
* a **latent diffusion model (LDM)** using **MONAI Generative**, operating in compressed latent space from **512×512** inputs

The conditioning signals used in both generative models are:

* **cell line label**
* **wound area percentage**

The scripts appear to support a proposal timeline roughly structured as:

* **Months 1–2**: data curation, preprocessing, augmentation, quality control
* **Months 3–4**: conditional generative model training and synthetic sample generation

---

## Overview

The repository currently includes code for four main stages:

1. **Step 1 — MONAI preprocessing pipeline**

   * loads paired microscopy images and masks
   * normalises and resizes data to **512×512**
   * infers **cell line labels** from filename prefixes
   * computes **wound area %** from masks
   * generates **3 augmentations per image**
   * saves processed arrays as `.npy`
   * writes metadata and visual summaries

2. **Preprocessing visualisation**

   * displays random processed image/mask pairs
   * overlays wound masks on microscopy images
   * checks that augmentations look structurally valid

3. **Step 4 — Conditional DDPM training**

   * trains a custom **conditional UNet denoiser**
   * uses **diffusion timestep + cell line embedding + wound area scalar** as conditioning
   * downsamples processed images from **512×512** to **256×256** for training
   * periodically generates synthetic wound images during training

4. **Step 5 — Latent diffusion model (LDM) with MONAI Generative**

   * trains an **AutoencoderKL** to compress images into a latent space
   * trains a **DiffusionModelUNet** in that latent space
   * uses cross-attention conditioning from **cell line + wound area %**
   * generates samples with **DDIM** for faster inference

---

## Dataset

The code is built for the **BBBC019** wound-healing assay dataset.

From the scripts:

* **165 paired images + ground-truth masks**
* after augmentation: **660 total samples**
* default preprocessing resolution: **512×512**
* DDPM training resolution: **256×256**

The preprocessing script assumes the following layout:

```text
/data/bbbc_019_clean/
├── images/
├── masks/
├── processed/
│   ├── images/
│   └── masks/
└── results/
    ├── preprocessing/
    ├── ddpm/
    └── ldm/
```

### Cell line mapping

Cell line IDs are inferred from filename prefixes:

| Prefix         | ID | Cell line |
| -------------- | -: | --------- |
| `SN15`         |  0 | DA3       |
| `Init`         |  0 | DA3       |
| `Melanoma`     |  1 | Melanoma  |
| `MDCK`         |  2 | MDCK      |
| `Microfluidic` |  2 | MDCK      |
| `Scatter`      |  2 | MDCK      |
| `HEK293`       |  3 | HEK293T   |
| `TScratch`     |  4 | Unknown   |

---

## Features

### Preprocessing

* supports `.tif`, `.tiff`, `.png`, `.bmp`, `.jpg`, `.jpeg`
* handles grayscale, RGB, and some multi-frame TIFF cases
* center-crops images to square before resizing
* normalises image intensities to `[0, 1]`
* converts masks to **wound = 1, cells = 0**
* computes **wound area percentage** from masks
* saves processed outputs as NumPy arrays for fast training

### Augmentation

The MONAI augmentation pipeline includes:

* random flips
* random 90° rotation
* random small-angle rotation
* Gaussian noise
* contrast adjustment
* zoom

Each original image produces:

* `1` original sample
* `3` augmented samples

for a total of `4×` samples per source image.

### Conditional DDPM

The custom image-space diffusion model includes:

* sinusoidal timestep embedding
* learned cell line embedding
* learned projection of wound area percentage
* conditional residual UNet blocks
* bottleneck self-attention
* periodic sample generation grids

### MONAI LDM

The latent diffusion pipeline includes:

* `AutoencoderKL` for image compression
* `DiffusionModelUNet` for latent denoising
* MONAI `DDPMScheduler` for training
* MONAI `DDIMScheduler` for faster sample generation
* cross-attention conditioning based on cell line and wound area

---

## Installation

Create an environment with Python 3.10+ and install the core dependencies.

```bash
pip install numpy pandas matplotlib pillow tifffile
pip install torch torchvision
pip install monai
pip install monai-generative
```

Depending on your environment, you may need a CUDA-enabled PyTorch build appropriate for your system.

### Main Python dependencies used in the scripts

* `torch`
* `numpy`
* `pandas`
* `matplotlib`
* `Pillow`
* `tifffile`
* `monai`
* `monai-generative`

---

## Hardware

The scripts are written for **GPU training on an NVIDIA A100**.

The code prints GPU metadata such as:

* GPU name
* total VRAM
* CUDA version
* PyTorch version

The preprocessing step also writes a GPU verification text file:

```text
results/gpu_verification.txt
```

Although the scripts target an A100, you may be able to run them on smaller GPUs by reducing:

* batch size
* image size
* model width
* number of workers

---

## Usage

## 1) Preprocess the dataset

Run the preprocessing pipeline to:

* pair images and masks
* resize them to `512×512`
* compute wound area percentages
* generate augmentations
* save metadata and visualisations

```bash
python step1_preprocessing.py
```

### Outputs

Generated files include:

```text
/data/bbbc_019_clean/processed/images/*.npy
/data/bbbc_019_clean/processed/masks/*.npy
/data/bbbc_019_clean/results/preprocessing/dataset_metadata.csv
/data/bbbc_019_clean/results/preprocessing/preprocessing_grid.png
/data/bbbc_019_clean/results/preprocessing/wound_area_distribution.png
/data/bbbc_019_clean/results/gpu_verification.txt
```

---

## 2) Visualise processed samples

This script samples processed `.npy` image/mask pairs and produces:

* random image/mask/overlay grids
* augmentation sanity checks

```bash
python visualize_processed.py
```

### Outputs

```text
/data/bbbc_019_clean/results/preprocessing/sample_processed_images.png
/data/bbbc_019_clean/results/preprocessing/augmentation_check.png
```

---

## 3) Train the conditional DDPM

The DDPM training script:

* loads processed images and metadata
* downsamples from `512×512` to `256×256`
* trains a conditional UNet denoiser
* saves the best checkpoint
* generates synthetic sample grids every 25 epochs

```bash
python step4_conditional_ddpm.py
```

### Default DDPM config

* image size: `256×256`
* batch size: `8`
* epochs: `200`
* learning rate: `2e-4`
* diffusion timesteps: `1000`
* number of cell lines: `5`

### DDPM outputs

```text
/data/bbbc_019_clean/results/ddpm/best_model_a100.pth
/data/bbbc_019_clean/results/ddpm/generated_epoch_XXXX.png
/data/bbbc_019_clean/results/ddpm/loss_curve.png
```

---

## 4) Train the latent diffusion model

The LDM workflow is a two-phase pipeline:

### Phase 1 — AutoencoderKL

* trains an autoencoder to compress `512×512` images to latent representations
* stores best and last checkpoints
* saves reconstruction grids during training

### Phase 2 — Latent diffusion training

* freezes the autoencoder
* trains a diffusion UNet in latent space
* conditions generation on cell line + wound area percentage
* periodically generates samples with DDIM

```bash
python step5_ldm_monai.py
```

### Default LDM config

* input size: `512×512`
* latent channels: `3`
* autoencoder epochs: `100`
* diffusion epochs: `200`
* diffusion timesteps: `1000`
* batch size: `4`

### LDM outputs

```text
/data/bbbc_019_clean/results/ldm/best_autoencoder.pth
/data/bbbc_019_clean/results/ldm/last_autoencoder.pth
/data/bbbc_019_clean/results/ldm/best_ldm.pth
/data/bbbc_019_clean/results/ldm/last_ldm.pth
/data/bbbc_019_clean/results/ldm/ae_recon_epoch_XXXX.png
/data/bbbc_019_clean/results/ldm/ae_loss_curve.png
/data/bbbc_019_clean/results/ldm/ldm_samples_epoch_XXXX.png
/data/bbbc_019_clean/results/ldm/ldm_loss_curve.png
```

---

## Model details

## Conditional DDPM

The DDPM implementation includes:

* **linear beta schedule**
* **1000 diffusion timesteps**
* **custom conditional UNet**
* **residual blocks with conditioning injection**
* **bottleneck self-attention**
* direct image-space denoising

Conditioning is fused from:

* timestep embedding
* cell line embedding
* wound percentage projection

This makes it possible to generate wound-healing images under specific experimental conditions.

## MONAI latent diffusion model

The LDM implementation includes:

* `AutoencoderKL`
* `DiffusionModelUNet`
* `LatentDiffusionInferer`
* `DDPMScheduler` for training
* `DDIMScheduler` for faster inference
* conditioning context generated from cell line and wound area percentage

This approach reduces memory cost by performing diffusion in latent space rather than pixel space.

---

## Conditioning scheme

Both generative approaches use the same two biological/experimental controls:

1. **Cell line**
2. **Wound area percentage**

This is intended to support controllable synthesis of microscopy images such as:

* small vs large wounds
* different cell-line morphologies
* interpolation across wound closure states

The sample-generation utilities in the scripts already include examples such as:

* DA3 with small and large wounds
* Melanoma at intermediate wound percentages
* MDCK and HEK293T examples

---

## Saved metadata

The preprocessing script writes a CSV file containing per-sample metadata, including:

* processed sample stem
* source filename
* cell line ID
* cell line name
* wound area percentage
* whether the sample is augmented

Expected file:

```text
/data/bbbc_019_clean/results/preprocessing/dataset_metadata.csv
```

This metadata is later used during generative model training.

---

## Notes and assumptions

A few implementation details are worth noting:

* the scripts assume the dataset already exists under `/data/bbbc_019_clean`
* cell-line labels are inferred from filename prefixes rather than a separate annotation table
* masks are interpreted as:

  * white = cell-covered region
  * black = wound region
* the DDPM script expects metadata at:

  * `/data/bbbc_019_clean/results/preprocessing/dataset_metadata.csv`
* the LDM script imports MONAI Generative modules using:

  * `from generative...`
* the visualisation script focuses on processed `.npy` files whose stems contain `_orig`
