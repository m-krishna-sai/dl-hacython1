# TRUSTMED — Comprehensive Final Project & Clinical Evaluation Report

**Project Title**: TRUSTMED — Self-Aware Multi-Specialty Medical AI for Reliable Disease Detection  
**Lead ML Engineer**: TRUSTMED AI Team  
**Evaluation Date**: September 12, 2026  
**Hardware Profile**: Intel Core i5-1334U (Laptop CPU), 12 GB RAM, Intel Iris Xe Integrated Graphics  
**Repository Policy**: Strictly Zero Synthetic Data in Final Metrics | Patient-Stratified Splits | Storage <= 7.0 GB  

---

## 1. Executive Summary
Most clinical medical AI systems answer only: *"What does this image show?"* TRUSTMED asks the critical second question: ***"How much should a clinician trust this prediction?"***

TRUSTMED is a multi-specialty AI decision-support platform spanning four critical clinical domains:
1. **🫁 Pulmonology (Chest Radiography)**: Multi-label thoracic disease classification across NIH ChestX-ray14 14 findings.
2. **🧠 Neuro-Oncology (Brain MRI)**: Voxel-level brain tumor segmentation (BraTS 2021) utilizing standard 2D U-Net with Dice + BCE loss.
3. **🩺 Dermatology (Dermoscopy)**: Multi-class skin lesion diagnosis across ISIC 2019 8 diagnostic classes.
4. **👁️ Ophthalmology (Retinal Fundus)**: Diabetic retinopathy severity grading across APTOS 2019 5 stages (Grades 0–4).

The platform integrates a **Trust Engine** that fuses Image Quality Gating, Specialty Auto-Routing, Calibrated Confidence (via validation Temperature Scaling), Monte Carlo Dropout Uncertainty (N=10), Out-of-Distribution (OOD) feature distance detection, Prediction Stability testing, and Grad-CAM visual interpretability into an actionable 3-tier Reliability Rating (**HIGH**, **MEDIUM**, **LOW**) with automated human specialist referral.

---

## 2. Problem Statement & Motivation
Medical AI models are notoriously susceptible to silent failures:
- **Overconfident errors**: Deep neural networks frequently produce >= 98% confidence on inputs that are blurry, corrupted, or out-of-distribution.
- **Spectrum bias and shortcut learning**: Models latch onto radiopaque markers, scanner borders, or institutional artifacts.
- **Data leakage in benchmarks**: Slices from the same patient or subject appearing across train and test sets create inflated, non-generalizable metric reports.

TRUSTMED addresses this failure mode by equipping models with **metacognitive self-awareness**—the ability to estimate and communicate the limits of their own diagnostic competence.

---

## 3. Dataset Storage & Acquisition Audit

Experiments were conducted on reproducible subsets of the original public datasets due to storage and computational constraints within an enforced **7.0 GB maximum storage ceiling**:

| Specialty | Dataset | Official Source | Full Original Size | Files Stored on Disk | Manifest Subset (Train / Val / Test) | Downloaded Storage | Purpose & Task |
|---|---|---|---:|---:|---|---:|---|
| **Lung** | NIH ChestX-ray14 | NIH Clinical Center / HF (`alkzar90`) | ~42.0 GB (112,120 scans) | 9,998 real images | 2,517 scans (1,610 / 417 / 490) across 697 patients | 2.02 GB | Multi-label Thoracic Classification (14 classes) |
| **Brain** | BraTS 2021 | RSNA-ASNR-MICCAI / FeTS (`Unknown6197`) | ~120.0 GB (3D NIfTI) | 5,900 slices (50 subjects) | 4,720 slices (3,304 / 708 / 708) across 40 subjects | 0.69 GB | Brain Tumor Segmentation (Dice + IoU) |
| **Skin** | ISIC 2019 | International Skin Imaging Collaboration | ~36.0 GB (25,331 images) | 25,331 real images | 3,038 images (2,124 / 452 / 462) across 8 classes | 0.15 GB | Dermoscopy Lesion Classification (8 classes) |
| **Retina** | APTOS 2019 | Aravind Eye Hospital / APTOS | ~8.2 GB (3,662 images) | 1,150 real images | 1,150 images (805 / 172 / 173) across 5 grades | 3.03 GB | Diabetic Retinopathy Grading (Grades 0–4) |
| **TOTAL** | | | **~206.2 GB** | **42,379 physical files** | **11,425 manifest entries / 40 subjects** | **5.89 GB** | **BUDGET STATUS: PASS (<= 7.0 GB)** |

*Count Verification*:
- **Files Physically Stored on Disk**: 42,379 files (9,998 Lung + 5,900 Brain + 25,331 Skin + 1,150 Retina).
- **Files in Experimental Manifests**: 11,425 entries (2,517 Lung + 4,720 Brain + 3,038 Skin + 1,150 Retina).
- **Files Evaluated on Held-Out Test Sets**: 1,833 test samples (490 Lung + 708 Brain + 462 Skin + 173 Retina).

---

## 4. Patient- & Subject-Level Split Policy (Zero Leakage)
To prevent data contamination, splitting was enforced strictly at the **Patient ID** (Lung) and **Subject ID** (Brain) levels using fixed random seed `SEED = 42`:
- **Lung**: 487 Train patients (1,610 scans), 104 Validation patients (417 scans), 106 Held-Out Test patients (490 scans). **Train/Test Overlap: 0 | Train/Val Overlap: 0 | Val/Test Overlap: 0**.
- **Brain**: 28 Train subjects (3,304 slices), 6 Validation subjects (708 slices), 6 Held-Out Test subjects (708 slices). **Train/Test Overlap: 0 | Train/Val Overlap: 0 | Val/Test Overlap: 0**.
- **Skin**: 2,124 Train, 452 Validation, 462 Held-Out Test images. **Leakage: 0**.
- **Retina**: 805 Train, 172 Validation, 173 Held-Out Test images. **Leakage: 0**.

---

## 5. Master Final Results Table

Every metric below is derived strictly from real predictions on patient/subject held-out test partitions:

| Specialty | Dataset | Evaluated Test Samples | Task | Model Architecture | Primary Metric | Sensitivity | Specificity | Dice | IoU | Calibration T |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| **Lung** | NIH ChestX-ray14 | 490 Held-Out Scans | Classification | ImageNet ResNet-18 | **Macro AUROC: 0.6552** | 0.0092 (th=0.50) | 0.9961 (th=0.50) | N/A | N/A | 1.1480 |
| **Brain** | BraTS 2021 | 708 Held-Out Slices (6 subjects) | Segmentation | Standard 2D U-Net | **Global Voxel Dice: 0.0473** | 0.0624 (Global) | 0.9872 (Global) | 0.0473 (Global) | 0.0242 (Global) | N/A |
| **Skin** | ISIC 2019 | 462 Held-Out Lesions | Classification | ImageNet ResNet-18 | **Macro AUROC: 0.8922** | 0.4324 | 0.9375 | N/A | N/A | 1.2940 |
| **Retina** | APTOS 2019 | 173 Held-Out Scans | Classification | ImageNet ResNet-18 | **Macro AUROC: 0.9061** | 0.6666 | 0.9244 | N/A | N/A | 1.3300 |

---

## 6. In-Depth Scientific Analysis & Methodological Transparency

### 6.1 Brain Tumor Segmentation Evaluation (Full 708-Slice Held-Out Test Set)
The Brain U-Net was evaluated on the **complete held-out test split of 708 slices** (from 6 unseen BraTS subjects):
* **Tumor-Bearing Slices ($N=364$, 51.4%)**: Mean Dice = **0.0241**, Mean IoU = **0.0134**, Mean Sensitivity = **0.0634**.
* **Global Dataset-Level Voxel Counts ($708 	imes 224 	imes 224$ voxels)**:
  * $TP = 17,815$, $FP = 450,211$, $FN = 267,764$, $TN = 34,788,818$
  * **Global Voxel Dice**: **0.0473**
  * **Global Voxel IoU**: **0.0242**
  * **Global Sensitivity**: **0.0624**
  * **Global Specificity**: **0.9872**
* **Methodological Finding on the 150-Slice Subset**:
  On an initial 150-slice subset, evaluating slice-level Dice with standard smoothing formula produced `0.7105`. Investigation revealed this was an artifact of 27 out of 38 batches having zero tumor in ground truth and zero predicted tumor, producing smoothed ratio $1.0$ ($27 / 38 = 0.7105$). In this final report, we transparently report the **Global Voxel Dice (0.0473)** and **Tumor-Bearing Slice Dice (0.0241)** on the complete 708-slice test set.

### 6.2 Lung ResNet-18 Threshold Analysis & Class Imbalance
The Lung model achieved a **Macro AUROC of 0.6552**, which is threshold-independent and measures true discrimination.
However, because NIH ChestX-ray14 has an empirical background disease prevalence of only **5.20%** (357 positive labels out of 6,860 evaluated targets), applying a generic binary decision threshold of `0.50` causes almost all predictions to fall below threshold (only 29 predictions reach >= 0.50).

To ensure complete scientific rigor, we report test performance across decision thresholds:

| Decision Threshold | Mean Sensitivity | Mean Specificity | Overall Accuracy | Clinical Interpretation |
|---|---:|---:|---:|---|
| **0.50 (Default Binary)** | **0.0092** | **0.9961** | **0.9449** | Near-zero false alarms; misses low-prevalence findings |
| **0.20** | **0.1261** | **0.9458** | **0.9112** | Conservative clinical operating point |
| **0.10** | **0.2587** | **0.8662** | **0.8516** | Balanced multi-label detection point |
| **0.05** | **0.4225** | **0.7190** | **0.7226** | Aligned with true 5.2% disease prevalence |
| **0.02 (Validation Youden J)** | **0.7426** | **0.4063** | **0.4364** | Optimal validation screening sensitivity point |

---

## 7. Optional Model Improvement: Attention Ablation Study
A controlled ablation experiment was conducted comparing ImageNet-pretrained ResNet-18 baseline against ResNet-18 augmented with CBAM (Convolutional Block Attention Module) on the exact same patient split:

| Model Architecture | Parameters | Multi-Label Thoracic AUROC | Change |
|---|---:|---:|:---:|
| **Standard ResNet-18 Baseline** | 11.2 Million | 0.6552 | Baseline |
| **ResNet-18 + CBAM Attention** | 11.4 Million | 0.6728 | +0.0176 Real Measured Improvement |

---

## 8. Trust Engine Architecture & Safety Modules

### 8.1 Image Quality Gate
Analyzes input scans prior to inference:
- Detects invalid image encodings or corrupt headers.
- Rejects images with severe blur (Laplacian variance < 40.0), extreme underexposure (mean intensity < 20), overexposure (> 235), or low resolution (< 128 x 128).
- Automatically blocks inference on inadequate scans to avoid garbage-in/garbage-out scenarios.

### 8.2 Specialty Auto-Routing
Extracts color space signatures, structural aspect ratios, and intensity distributions:
- Correctly routes Chest X-rays, Brain MRIs, Dermoscopy images, and Retinal fundus photographs.
- **Unsupported Image Handling**: Out-of-scope modalities (e.g. panoramic dental X-rays, histopathology, ultrasound) are cleanly rejected with a clear clinical explanation.

### 8.3 Monte Carlo Dropout Uncertainty
Executes N=10 stochastic forward passes with dropout (p=0.3) active at test time:
- Computes predictive variance across passes: sigma^2 = (1/N) * sum((p_i - p_mean)^2).
- Computes predictive entropy: H(p) = -sum(p_c * log(p_c)).
- Returns normalized uncertainty score [0, 1].

### 8.4 Out-of-Distribution (OOD) Detection
- Extracts 512-dimensional penultimate feature vectors from `model.forward_features(x)`.
- Measures cosine distance to training class centroid prototypes.
- Flags scans exceeding the 95th percentile distance threshold as **POSSIBLY OUT-OF-DISTRIBUTION**.

### 8.5 Temperature Scaling Confidence Calibration
- Optimizes post-hoc temperature parameter T > 0 on validation logits via NLL minimization:
  - Lung: T = 1.148
  - Skin: T = 1.294
  - Retina: T = 1.330
- Preserves accuracy and rank ordering while aligning softmax probability with empirical correctness.

### 8.6 Prediction Stability Testing
- Perturbs input image through subtle micro-transformations (slight zoom, mild rotation +/- 2 degrees).
- Measures output prediction agreement across iterations.
- Classifies response as **Stable**, **Moderately Stable**, or **Unstable**.

### 8.7 Grad-CAM Explainability
- Generates Class Activation Maps from layer4 feature maps weighted by target class gradients.
- Visualizes heatmaps overlaid on the original clinical image to highlight regions influencing the prediction.

### 8.8 Reliability Assessment Engine
Synthesizes calibrated confidence, uncertainty, OOD status, stability, and image quality into an engineering risk tier:
- **HIGH RELIABILITY**: High confidence, low uncertainty, in-distribution, stable, good quality.
- **MEDIUM RELIABILITY**: Moderate confidence or borderline OOD/uncertainty.
- **LOW RELIABILITY**: High uncertainty, OOD, unstable, or low quality.

> [!WARNING]
> **High-Confidence / Low-Reliability Case**: If an out-of-distribution scan produces high softmax probability (e.g. 96%) but exhibits high MC dropout variance or high feature prototype distance, TRUSTMED overrides the confidence and assigns **LOW RELIABILITY**, recommending urgent human specialist review.

---

## 9. Real-Time Hardware Performance Benchmark

Executed natively on **Intel Core i5-1334U CPU (12 GB RAM)** from `reports/benchmark_report.json`:

| Metric | Measured Value | Throughput | Performance Standard |
|---|---:|---:|:---:|
| **Startup Model Preloading Latency** | 953.8 ms | — | Sub-2 seconds |
| **Single-Scan End-to-End Pipeline Latency** | 415.7 ms | 2.40 scans/s | Real-time interactive (< 500 ms) |
| **Batch 1 Pipeline Latency** | 416.7 ms | 2.40 scans/s | Interactive |
| **Batch 2 Pipeline Latency** | 746.6 ms (373.3 ms/scan) | 2.68 scans/s | Efficient multi-image inference |
| **Batch 3 Pipeline Latency** | 1154.4 ms (384.8 ms/scan) | 2.60 scans/s | Linear CPU throughput |
| **Batch 4 Pipeline Latency** | 1559.8 ms (389.9 ms/scan) | 2.56 scans/s | Stable parallel processing |
| **Active Memory Footprint (RAM RSS)** | 590.2 MB | — | Ultra-compact (< 1 GB) |

---

## 10. Regulatory & Ethical Disclaimer
**TRUSTMED is an engineering decision-support research prototype.**  
It is **NOT** a certified medical device, does not provide medical prescriptions, diagnoses, or clinical treatment recommendations, and does not replace the clinical judgment of a licensed healthcare professional. All algorithmic outputs must be reviewed by qualified human specialists.
