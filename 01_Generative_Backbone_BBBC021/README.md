# 01_Generative_Backbone_BBBC021

Foundational validation of generative architectures including cVAE, cGAN using the BBBC021 benchmark dataset.

---

# BBBC021 Results — cVAE / cGAN for Cell Painting Patch Generation

This repository contains experiments on **generative modeling for microscopy image patches** from the **BBBC021** Cell Painting dataset, with a focus on **conditional VAEs (cVAEs)** and a comparison against a **conditional GAN (cGAN)** baseline.

The goal of the project is to generate biologically plausible image patches conditioned on **mechanism of action (MoA)** categories, and to identify a model configuration that is both quantitatively strong and visually useful for presentation-quality results.

---

## Project goals

The project was designed around four practical objectives:

1. **Generate realistic multi-channel microscopy patches** from BBBC021.
2. **Condition generation on MoA class** so different biological categories can be sampled separately.
3. **Compare multiple generative configurations** in a systematic way.
4. **Select a final model** that balances:

   * reconstruction quality,
   * sample sharpness,
   * biological plausibility,
   * and poster/report readability.

Because the project is deadline-driven, the emphasis is on **small, high-impact experimental changes** rather than repeated full architectural rewrites.

---

## Dataset

We use **BBBC021**, a Broad Bioimage Benchmark Collection dataset based on **MCF-7 human cells** treated with compounds from multiple mechanism-of-action classes.

* Original dataset page: BBBC021, compound-profiling experiment
* Image modality: multi-channel fluorescence microscopy
* Working representation in this repository: **256×256 image patches saved as `.pt` tensors**

The data pipeline assumes a patch-based structure similar to:

```text
/data/annapan/prop/bbbc021/patches_256_pt/train/<MoA name>/imgXXXXX_patchYY.pt
```

A CSV metadata file is used to define train/validation splits and labels.

### MoA classes

The project uses mapped MoA labels such as:

* Actin disruptors
* Aurora kinase inhibitors
* Cholesterol-lowering
* DMSO
* DNA damage
* DNA replication
* Eg5 inhibitors
* Epithelial
* Kinase inhibitors
* Microtubule destabilizers
* Microtubule stabilizers
* Protein degradation / synthesis-related classes

Exact label mappings are stored in training checkpoints.

---

## Methods

### 1. Conditional VAE

The main model family is a **conditional variational autoencoder (cVAE)**.

#### Why cVAE?

A cVAE was chosen because it provides:

* stable training,
* controllable class-conditional generation,
* meaningful latent representations,
* and reconstruction-based evaluation, which is especially useful for microscopy images.

#### Base objective

Most runs optimize:

```math
\mathcal{L} = \text{Recon}(x, \hat{x}) + \beta \, KL(q(z|x,y) \| p(z|y))
```

where:

* `Recon` is primarily **L1 reconstruction loss**,
* `beta` controls latent regularization,
* `y` is the class condition (MoA label).

### 2. Stronger conditioning variant

A second architecture variant, referred to as **cVAE-condv2**, introduces **stronger conditioning** inside the decoder to encourage more class-specific generation.

This was tested after identifying a promising beta value from the base cVAE experiments.

### 3. Cosine learning rate scheduler

A cheap but effective refinement was to add:

* **CosineAnnealingLR**

This was introduced to improve late-stage optimization and stabilize the final epochs.

### 4. Latent dimension scaling

The latent dimension was increased from:

* **128 → 256**

to test whether additional latent capacity improves reconstruction detail and sample quality.

### 5. MS-SSIM mixed loss

One experiment replaced the plain reconstruction term with a mixture of:

* **0.84 × MS-SSIM loss + 0.16 × L1 loss**

This was motivated by the idea that structural losses may preserve texture and larger-scale organization better than pure pixelwise loss.

### 6. cGAN baseline

A **conditional GAN** was also trained for comparison.

This was not the main direction of the project, but it was useful as a second model family for qualitative comparison.

---

## Experimental roadmap

The experiments followed a staged plan.

### Phase 1 — beta ablation

These runs kept everything fixed except `beta`.

* **Run A**: baseline cVAE, `beta = 1e-3`
* **Run B**: cVAE, `beta = 5e-4`
* **Run C**: cVAE, `beta = 2e-3`

### Phase 2 — follow-ups from the best beta

Using the best beta from Phase 1:

* **Run D**: stronger conditioning (`cVAE-condv2`)
* **Run E**: cosine scheduler added to the base cVAE
* **Run F**: cosine scheduler + larger latent dimension (`latent_dim = 256`)
* **Run G**: cosine scheduler + MS-SSIM mixed loss
* **Run H**: stronger conditioning + cosine + latent 256
* **Run I**: cGAN comparison baseline

---

## Summary of key runs

Below is the final high-level summary used to track the project.

| Run | Model       | Key setting                                 | Main outcome                                      |
| --- | ----------- | ------------------------------------------- | ------------------------------------------------- |
| A   | cVAE        | baseline, beta=1e-3                         | Good baseline, smooth samples                     |
| B   | cVAE        | beta=5e-4                                   | Better than A, cleaner and sharper                |
| C   | cVAE        | beta=2e-3                                   | Stronger regularization, but visually weaker      |
| D   | cVAE-condv2 | stronger conditioning                       | Close numerically, not visually better than B     |
| E   | cVAE        | B + cosine scheduler                        | Slight but real improvement over B                |
| F   | cVAE        | E + latent_dim=256                          | **Best overall model**                            |
| G   | cVAE        | cosine + MS-SSIM mixed loss                 | Preserved structure, but reconstructions too soft |
| H   | cVAE-condv2 | stronger conditioning + latent 256 + cosine | Strong numerically, but still softer than F       |
| I   | cGAN        | conditional GAN baseline                    | Mode collapse; inferior to cVAE                   |

---

## Final results

### Best model

The strongest final model in this project is:

**Run F**

* Model: `cVAE`
* `beta = 5e-4`
* `latent_dim = 256`
* `epochs = 120`
* scheduler: `CosineAnnealingLR`
* reconstruction loss: `L1 + KL`

### Best recorded metrics for Run F

* **best epoch:** 116
* **best val_loss:** 0.0244
* **best val_recon:** 0.0231
* **best val_kl:** 2.5398

### Why Run F won

Run F gave the best overall tradeoff between:

* sharper reconstructions,
* stronger sample quality,
* plausible cell morphology,
* and clean presentation quality.

It clearly improved over earlier cVAE runs and remained more stable and useful than the cGAN baseline.

---

## Qualitative findings

### cVAE family

Main observations across the cVAE runs:

* Lower `beta` values improved sharpness.
* Adding a cosine scheduler gave a small but consistent gain.
* Increasing latent dimension to 256 gave the most important improvement.
* Stronger conditioning helped numerically, but did not produce a clearly superior final visual result.
* MS-SSIM preserved coarse image structure, but outputs remained too soft for the final choice.

### cGAN baseline

The cGAN produced sharper-looking motifs in some cases, but the final comparison showed **mode collapse**:

* within-class samples became nearly identical,
* diversity was poor,
* and the model was not reliable enough for the final result.

This comparison ultimately supports the choice of the **cVAE** as the main model family for the project.

---

## Repository structure

A typical structure for this repository is:

```text
.
01_Generative_Backbone_BBBC021/
├── BBBC021_v1_compound.csv
├── BBBC021_v1_image.csv
├── BBBC021_v1_moa.csv
├── bbbc021_moa_resolved.csv
├── bbbc021_moa_subset.csv
├── file_index.csv
├── folds_by_compound.json
├── moa_to_idx.json
├── patches_256_metadata.csv
├── split_80_20_row_stratified.json
├── split_80_20_stratified_grouped.json
├── results/
│   ├── README.md
│   ├── cVAE_cGAN_experiment_results.csv
│   └── cVAE_cGAN_experiment_results.xlsx
└── scripts/
    ├── build_file_index.py
    ├── build_folds.py
    ├── build_moa_resolved.py
    ├── build_moa_subset.py
    ├── build_split.py
    ├── build_split_row_stratified.py
    ├── dataset_bbbc021.py
    ├── dataset_bbbc021_patches.py
    ├── make_patches_pt.py
    ├── moa_label_mapping.py
    ├── sample_cgan.py
    ├── sample_cvae.py
    ├── sample_cvae_condv2_from_run.py
    ├── sample_cvae_from_run.py
    ├── sample_cvae_v2.py
    ├── save_recon_panel_condv2_from_run.py
    ├── save_recon_panel_from_run.py
    ├── test_dataset.py
    ├── test_patches.py
    ├── train_cgan.py
    ├── train_cvae.py
    ├── train_cvae_condv2.py
    ├── train_cvae_condv2_cosine.py
    ├── train_cvae_exp.py
    ├── train_cvae_exp_cosine.py
    ├── train_cvae_exp_msssim.py
    ├── train_cvae_v2.py
    ├── visualize_tsne.py
    └── model/
        ├── model_cgan.py
        ├── model_cvae.py
        └── model_cvae_condv2.py
```

---

## Training setup

Common settings used across the cVAE runs:

* patch size: **256 × 256**
* batch size: **32**
* learning rate: **1e-4**
* optimizer: **Adam**
* augmentation: **flip + 90° rotations**
* weighted sampler: **enabled**
* number of epochs: **80–120** depending on run

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cvae_exp_cosine.py \
  --run_name cvae_patch256_ld256_beta5e-4_bs32_augfliprot_e80_cosine \
  --beta 5e-4 \
  --epochs 120 \
  --latent_dim 256 \
  --cond_dim 32 \
  --batch_size 32 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --weighted_sampler
```

---

## Evaluation workflow

The project evaluates models using both **quantitative** and **qualitative** criteria.

### Quantitative metrics

For VAE-based runs:

* validation loss
* validation reconstruction loss
* validation KL divergence

### Qualitative outputs

For each run, the following visual outputs are typically generated:

* **class sample panels**
* **reconstruction panels**
* optional **latent t-SNE plots**

### Visual scoring criteria

Each run can be scored on:

* reconstruction sharpness
* sample sharpness
* class distinctiveness
* biological plausibility
* poster readability

This was important because the final selection was **not based on val loss alone**.

---

## Main conclusion

The project supports the following conclusion:

> A conditional VAE is a strong and stable solution for class-conditional BBBC021 patch generation, and the best-performing setup in this repository is a cVAE with **beta = 5e-4**, **cosine annealing**, and **latent dimension 256**.

In practice, this means that **Run F** is the best final model from the current study.

---

## Limitations

Current limitations of the work include:

* class separation in the latent space is still limited,
* generated samples are biologically plausible but not perfectly sharp,
* patch-based generation ignores larger field-of-view structure,
* cGAN training was unstable and collapsed,
* no external downstream biological evaluation was included yet.

---

## Possible future work

Good next steps would include:

* per-class quantitative evaluation,
* FID-style or embedding-based biological similarity metrics,
* improved conditioning mechanisms,
* higher-capacity latent priors,
* diffusion-based baselines,
* full-image generation beyond patches,
* and downstream MoA classification using generated data augmentation.

---

## Reproducibility notes

To reproduce results, keep track of:

* run folder name,
* model type,
* beta,
* latent dimension,
* scheduler,
* loss definition,
* conditioning variant,
* augmentation setup,
* and whether weighted sampling was enabled.

The repository includes experiment tracking scripts and spreadsheets for this purpose.

---

## Acknowledgments

* **BBBC021** from the Broad Bioimage Benchmark Collection
* This work was supported through access to NVIDIA A100 Tensor Core GPUs provided by the NVIDIA Hardware Grant Program. We gratefully acknowledge NVIDIA for supporting the computational infrastructure used in this project.
---



