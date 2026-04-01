# Conditional DDPM for Simulated Wound Trajectories

This folder contains a single training script for a **conditional denoising diffusion probabilistic model (DDPM)** designed to generate microscopy-like cell images conditioned on class labels.

The implementation uses a **U-Net backbone**, **classifier-free guidance (CFG)**, and a full training / validation / sampling pipeline in one file.

## File

- `conditional_ddpm.py` — end-to-end script for dataset loading, model definition, training, checkpointing, and sample generation

## Overview

The script implements an all-in-one conditional DDPM pipeline for multi-channel cell imaging data.

### Main components
- TIFF-based dataset loader
- Multi-channel image preprocessing
- Conditional U-Net denoiser
- Diffusion forward and reverse process
- Class-conditioned image generation with classifier-free guidance
- Training, validation, checkpoint saving, and sample export
- Smoke-test mode for quick debugging

## Input Data

The script expects:
- a CSV index file with image paths and class labels
- three fluorescence channels per sample:
  - DAPI
  - Tubulin
  - Actin

Images are resized to `64 × 64` and normalized to `[-1, 1]`.

## Model

The model includes:
- **Conditional U-Net** backbone
- **Sinusoidal timestep embeddings**
- **Class embeddings**
- **Residual blocks with conditioning**
- **Self-attention layers**
- **Classifier-free guidance (CFG)** during sampling

## Training

Training includes:
- random train/validation split
- MSE noise-prediction objective
- AdamW optimizer
- cosine annealing learning-rate scheduler
- gradient clipping
- periodic checkpointing
- periodic sample generation

## Outputs

During training, the script saves:

- **Checkpoints** to the configured checkpoint directory
- **Generated sample grids** to the configured sample directory
- **Best-performing model** as `ddpm_best.pt`

## Usage

Train from scratch:

```bash
python conditional_ddpm.py