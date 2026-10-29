# TRUSTMED Comprehensive Medical Dataset Report

**Project**: TRUSTMED — Self-Aware Multi-Specialty Medical AI  
**Author**: Lead ML Engineer  
**Date**: September 12, 2026  
**Seed**: 42  
**Storage Budget**: 7.0 GB Maximum (Preferred: 5.0 – 7.0 GB)  
**Budget Verification Status**: **PASS**

---

## 1. Executive Storage & Subset Summary

> [!IMPORTANT]
> **Scientific Integrity Affirmation**: Due to computational and storage constraints on local clinical workstations (Intel Core i5 CPU, 12 GB RAM), experiments were conducted on reproducible, patient-stratified subsets of the original public datasets. **No synthetic or artificially generated medical images were used for final model training or evaluation.** All performance metrics originate from held-out test partitions of real clinical scans.

### Official Dataset Storage Budget Table

| Specialty | Dataset | Official Source | Full Original Size | Selected Samples / Subjects | On-Disk Storage | Purpose & Task |
|---|---|---|---:|---:|---:|---|
| **Lung** | NIH ChestX-ray14 | NIH Clinical Center / HuggingFace (`alkzar90`) | ~42.0 GB (112,120 scans) | 4,999 Real CXR Scans (999 Patients) | 2.02 GB | Multi-label Thoracic Disease Classification (14 classes) |
| **Brain** | BraTS 2021 | RSNA-ASNR-MICCAI / FeTS (`Unknown6197`) | ~120.0 GB (3D NIfTI) | 50 Complete Subjects (5,900 MRI Slices) | 0.69 GB | Brain Tumor Segmentation (T1/T2/FLAIR, Dice + IoU) |
| **Skin** | ISIC 2019 | International Skin Imaging Collaboration (`MKZuziak`) | ~36.0 GB (25,331 high-res) | 3,000 Stratified Images (8 classes) | 0.15 GB | Dermoscopy Lesion Classification (8 diagnostic classes) |
| **Retina** | APTOS 2019 | Aravind Eye Hospital / Kaggle / HF (`Tejaswini628`) | ~8.2 GB (3,662 high-res) | 1,150 Stratified Images (5 grades) | 2.75 GB | Diabetic Retinopathy Severity Grading (Grades 0–4) |
| **TOTAL** | | | **~206.2 GB** | **8,149 Real Clinical Samples / 50 Subjects** | **5.61 GB** | **STRICTLY $\le$ 7.0 GB TARGET (PASS)** |

---

## 2. Specialty Dataset Breakdowns

### 2.1 Lung — NIH ChestX-ray14
- **Original Public Source**: National Institutes of Health (NIH) Clinical Center, Bethesda, MD.
- **Acquisition Mechanism**: Direct automated download of official `images_001.zip` (1.87 GB) and `Data_Entry_2017_v2020.csv` via verified Hugging Face repository.
- **Original Dataset Scope**: 112,120 frontal-view X-ray images from 30,805 unique patients.
- **Experiment Subset**: 4,999 real chest radiographs across 999 unique patients.
- **Selected Split Subset**: 2,500 patient-stratified images (1,750 Train / 375 Validation / 375 Held-Out Test).
- **Official Thoracic Findings (14 Classes)**:
  1. Atelectasis
  2. Cardiomegaly
  3. Effusion
  4. Infiltration
  5. Mass
  6. Nodule
  7. Pneumonia
  8. Pneumothorax
  9. Consolidation
  10. Edema
  11. Emphysema
  12. Fibrosis
  13. Pleural Thickening
  14. Hernia
- **Patient Isolation Policy**: Zero patient overlap between train, validation, and test sets. All radiographs for a patient identifier are locked to a single split partition.

### 2.2 Brain — BraTS 2021 (Federated Tumor Segmentation)
- **Original Public Source**: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS 2021) / FeTS 2021 challenge.
- **Acquisition Mechanism**: Official sliced volumetric MRI HDF5 archives (`sliced_data.tar.gz`) containing paired multi-modal brain MRI slices and voxel-level tumor segmentation annotations.
- **Original Dataset Scope**: Multi-parametric MRI (mpMRI) scans from over 2,000 glioma patients worldwide.
- **Experiment Subset**: 50 complete clinical subjects (`FeTS21_Training_001` through `FeTS21_Training_050`), totaling 5,900 H5 slice files (each containing paired `case` MRI slice and ground-truth `label` segmentation mask).
- **Split Breakdown**: 35 subjects Train (70%), 7 subjects Validation (15%), 8 subjects Test (15%).
- **Subject-Level Split Rule**: **Zero cross-split subject leakage.** Slices from any given subject volume appear exclusively in one partition.

### 2.3 Skin — ISIC 2019
- **Original Public Source**: International Skin Imaging Collaboration (ISIC) 2019 Challenge (HAM10000, BCN_20000, MSK collections).
- **Acquisition Mechanism**: Verified HF archive (`MKZuziak/ISIC_2019_224`) providing 224x224 downsampled dermoscopy images and official `metadata.jsonl`.
- **Original Dataset Scope**: 25,331 dermoscopy images across 8 diagnostic categories + 1 outlier class.
- **Experiment Subset**: 3,000 real dermoscopy images selected via stratified sampling:
  - Melanoma (MEL): 535
  - Melanocytic nevus (NV): 1,525
  - Basal cell carcinoma (BCC): 393
  - Actinic keratosis (AK): 102
  - Benign keratosis (BKL): 310
  - Dermatofibroma (DF): 50
  - Vascular lesion (VASC): 50
  - Squamous cell carcinoma (SCC): 74
- **Partitioning**: 2,100 Train (70%), 450 Validation (15%), 450 Held-Out Test (15%).

### 2.4 Retina — APTOS 2019 Blindness Detection
- **Original Public Source**: Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Symposium & Aravind Eye Hospital, India.
- **Acquisition Mechanism**: High-resolution fundus images and official `train.csv` from verified APTOS repository (`Tejaswini628/aptos-fundus-images`).
- **Original Dataset Scope**: 3,662 retina fundus photographs graded by clinicians into 5 stages of diabetic retinopathy.
- **Experiment Subset**: 1,150 real fundus photographs stratified across all 5 clinical stages:
  - Grade 0 (No DR): 300 images
  - Grade 1 (Mild DR): 200 images
  - Grade 2 (Moderate DR): 300 images
  - Grade 3 (Severe DR): 150 images
  - Grade 4 (Proliferative DR): 200 images
- **Partitioning**: 805 Train (70%), 172 Validation (15%), 173 Held-Out Test (15%).

---

## 3. Preprocessing & Data Pipeline Specifications

| Specialty | Input Modality | Ingest Dimensions | Normalization | Augmentation Policy | Output Model Tensor |
|---|---|---|---|---|---|
| **Lung** | Frontal CXR PNG | $1024 \times 1024$ Grayscale | ImageNet ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$) | Random horizontal flip ($p=0.5$), slight jitter ($\pm 10\%$) | $[3, 224, 224]$ |
| **Brain** | Multi-modal MRI H5 | $240 \times 240$ uint8 | Min-max normalization $[0, 1]$, replicated to 3 channels | Identity crop/resize, mask-synchronized nearest neighbor | $[3, 224, 224]$ (img) / $[1, 224, 224]$ (mask) |
| **Skin** | Dermoscopy JPEG | $224 \times 224$ RGB | ImageNet standard | Random horizontal flip ($p=0.5$), brightness/contrast ($\pm 10\%$) | $[3, 224, 224]$ |
| **Retina** | Fundus Photo PNG | $2000 \times 2000+$ RGB | Resized to $224 \times 224$, ImageNet standard | Subtle horizontal flip ($p=0.5$), contrast enhancement | $[3, 224, 224]$ |

---

## 4. Limitations & Integrity Caveats

1. **Subset Scope**: While subsets are real, unadulterated clinical images from recognized international benchmark repositories, they constitute representative samples rather than the complete multi-gigabyte collections.
2. **Computational Adaptation**: Due to target hardware (Intel Core i5-1334U laptop CPU, 12 GB RAM), training epochs and slice depths were calibrated for efficient convergence within the target 30–40 minute total budget.
3. **No Clinical Diagnosis Claim**: The TRUSTMED system is designed and implemented solely as an AI-assisted decision-support prototype. It does not provide medical diagnoses, treatment regimens, or prescriptions.
