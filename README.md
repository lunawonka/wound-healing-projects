# Generative and Multimodal Deep Learning for Simulated Wound Healing

Executed on NVIDIA A100 Tensor Core GPUs through the NVIDIA Hardware Grant Program.

## Project Overview

This repository develops a hybrid computational-experimental framework for simulating and forecasting wound healing trajectories. By combining conditional generative models, latent diffusion methods, and radiomic analysis, the project aims to establish a **synthetic experimental platform** for exploring healing dynamics under different biological and treatment conditions.

The long-term goal is to support **in silico what-if experimentation** for wound healing, reduce the cost of repeated in vitro screening, and improve the optimization of Photobiomodulation (PBM) therapy.

## Scientific Motivation

Wound healing is a dynamic and highly heterogeneous biological process. Experimental assays are informative, but often expensive, time-consuming, and difficult to scale across many treatment combinations. Our approach combines:

- **Generative deep learning**, to synthesize biologically plausible microscopy trajectories
- **Radiomics and feature-based modeling**, to identify structural biomarkers associated with healing progression
- **Multimodal conditioning**, to eventually link image evolution with treatment-specific metadata

Together, these components support a computational framework for predictive and interpretable regenerative medicine research.

## Repository Structure
```bash
├── 01_Generative_Backbone_BBBC021/
├── 02_Radiomics_Analysis_WoundHealing/
├── 03_Simulated_Wound_Trajectories/
├── 04_Scratch_Assay_LDM_Pipeline/
├── scratch_assay_ap_1st/
├── prop_BBBC021/
├── visualizations/
├── README.md
└── .gitignore
```

01_Generative_Backbone_BBBC021

Foundational validation of generative architectures including cVAE, cGAN, and DDPM using the BBBC021 benchmark dataset.

02_Radiomics_Analysis_WoundHealing

Radiomic feature extraction and downstream machine learning pipelines for NCTC keratinocyte scratch assays, including predictive modeling with methods such as XGBoost, SVR, and Cellpose-SAM-assisted segmentation.

03_Simulated_Wound_Trajectories

Implementation of Conditional DDPM with Classifier-Free Guidance (CFG) for temporal microscopy image synthesis and biologically guided wound trajectory simulation.

04_Scratch_Assay_LDM_Pipeline

Latent Diffusion Model (LDM) workflow for efficient and higher-resolution scratch assay simulation, including extensions motivated by BBBC019 and related microscopy benchmarks.



**Summary of Key Accomplishments**
Established Generative Foundation: We implemented a custom Conditional DDPM architecture, utilizing the NVIDIA NeMo Framework guidelines for scalable generative workflows. This model serves as the project's "simulation engine," capable of synthesizing high-resolution 3-channel fluorescence microscopy images. The use of Classifier-Free Guidance (CFG) allows for precise structural generation based on cell-line metadata.

Multimodal Metadata Integration & Feature Profiling: Leveraging NVIDIA-accelerated XGBoost and SVR, we conducted a systematic analysis of NCTC keratinocytes by integrating radiomic image features with experimental metadata (light dose, wavelength, and curcumin concentration). We identified dominant geometric "signatures"—specifically Sphericity and Perimeter-Surface Ratio—that act as quantitative predictors of therapeutic success.

Pipeline Scalability via MONAI: We utilized the MONAI framework for GPU-accelerated preprocessing of the BBBC019 and private PBM datasets. The A100’s high VRAM enabled the deployment of Cellpose-SAM for high-throughput instance segmentation, processing thousands of cell masks in parallel—a workflow that was computationally inaccessible on standard local hardware.


