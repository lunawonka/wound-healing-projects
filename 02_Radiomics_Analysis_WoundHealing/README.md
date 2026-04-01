# Module 02: Radiomics-Based Predictive Modeling of Wound Healing

## Overview

This module quantifies wound healing dynamics in NCTC keratinocytes treated with blue-light photodynamic therapy (PDT) and curcumin. Using a machine learning-based radiomics pipeline, it extracts morphological and textural signatures from time-lapse microscopy images to model healing progression and treatment response.

These radiomic features also serve as conditioning inputs for the generative framework developed in Module 03.

## Experimental Setup

- **Assay:** In vitro scratch assay  
- **Treatment:** 425 nm blue light + curcumin photosensitizer  
- **Imaging timepoints:** 0, 24, 48, and 72 hours post-irradiation  

## Pipeline

1. **Wound Mask Extraction**  
   Preprocessing with TV-Chambolle denoising, CLAHE, and rank entropy filtering for wound boundary detection.

2. **Cell Segmentation**  
   Instance segmentation using Cellpose-SAM.

3. **Radiomic Feature Extraction**  
   PyRadiomics-based extraction of >100 features, including:
   - Shape-based features
   - First-order statistics
   - Texture features (GLCM, GLRLM, NGTDM)

4. **Predictive Modeling**  
   Benchmarking of:
   - SVR
   - Random Forest
   - XGBoost

## Key Results

- **Top predictive features:** Geometric Sphericity, Perimeter-Surface Ratio  
- **Best classifier:** Random Forest  
- **Classification performance:** ROC-AUC = 0.786  
- **Biological trend:** Successful healing is associated with greater geometric compactness and lower textural non-uniformity, especially at 48 hours.

## Generative Modeling Link

The significant radiomic features identified could be used as conditioning variables for the DDPM model in Module 03.

Planned applications include:
- Latent conditioning using 25 key radiomic features
- Visual counterfactual generation under untested treatment settings
- Automated validation of synthetic images through the same radiomics pipeline

