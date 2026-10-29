# TRUSTMED — Self-Aware Multi-Specialty Medical AI

> **“Most medical AI systems answer: *What does this image show?* TrustMed asks a second question: *How much should we trust the prediction?*”**

![TrustMed Banner](https://img.shields.io/badge/TrustMed-v1.0.0-0284c7?style=for-the-badge)
![Track 2](https://img.shields.io/badge/Hackathon-Track%202:%20Deep%20Learning%20for%20Healthcare-06b6d4?style=for-the-badge)
![CPU Compatible](https://img.shields.io/badge/Platform-Laptop%20CPU%20(Intel%20i5)-10b981?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic%20Demo-8b5cf6?style=for-the-badge)

---

## 1. Executive Summary & Core Innovation

Traditional deep learning classifiers produce raw softmax probability values that are frequently conflated with true clinical confidence. When presented with out-of-distribution artifacts, novel pathologies, or blurry scans, standard models frequently output **95%+ confidence on incorrect diagnoses**.

**TrustMed** addresses this safety crisis by acting as a **self-aware reliability layer** across 4 critical specialties:
1. 🫁 **Lung (NIH ChestX-ray14)**: Multi-label thoracic disease classification with ResNet-18 & Grad-CAM.
2. 🧠 **Brain (BraTS 2021)**: Brain tumor abnormality segmentation with lightweight 2D U-Net + screening classifier.
3. 🩺 **Skin (ISIC 2019)**: 8-class dermoscopic lesion classification with ResNet-18 & Grad-CAM.
4. 👁️ **Retina (APTOS 2019)**: Diabetic retinopathy grading (0–4) with ResNet-18 & Grad-CAM.

---

## 2. System Architecture

```text
                           TRUSTMED ARCHITECTURE
                                     │
                                     ▼
                            [ IMAGE UPLOAD ]
                          (1 to 4 Images Batch)
                                     │
                                     ▼
                        [ IMAGE QUALITY GATE ]
                    (Laplacian Blur, Contrast, Range)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
                 [ GOOD ]                         [ POOR ]
                     │                               │
                     ▼                               ▼
          [ SPECIALTY AUTO-ROUTER ]            [ SAFE REJECTION ]
       (Modality Structure & Features)      (Prompt Clearer Scan)
                     │
       ┌─────────────┼─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
    [ LUNG ]      [ BRAIN ]     [ SKIN ]     [ RETINA ]
    ResNet-18      U-Net +     ResNet-18     ResNet-18
   Multi-Label   Segmentation   8 Classes     5 Grades
       │             │             │             │
       └─────────────┼─────────────┴─────────────┘
                     │
                     ▼
          [ TRUST ENGINE AUDIT ]
   ┌─────────────────┼─────────────────┐
   ▼                 ▼                 ▼
[ MC DROPOUT ]  [ FREE-ENERGY OOD ] [ STABILITY ]
(Epistemic Var) (Prototype Dist)    (Modal Agreement)
   │                 │                 │
   └─────────────────┼─────────────────┘
                     ▼
       [ TEMPERATURE CALIBRATION ]
                     │
                     ▼
       [ GRAD-CAM / MASK OVERLAYS ]
                     │
                     ▼
         [ RELIABILITY SYNTHESIS ]
      (HIGH / MEDIUM / LOW Verdict)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  [ HIGH RELIABILITY ]   [ LOW RELIABILITY ]
  AI-Assisted Review     URGENT HUMAN SPECIALIST REVIEW
                         (Flagged for Physician)
```

---

## 3. The 5 Pillars of the Trust Engine

| Pillar | Technical Implementation | Purpose |
|---|---|---|
| **1. Image Quality Gate** | Laplacian variance ($\sigma^2 \ge 40.0$), RMS contrast ($\ge 0.10$), exposure limits | Hard-gate rejection of corrupted, blurry, or washed-out images before inference. |
| **2. Modality Auto-Router** | Geometric, aspect-ratio, border darkness & color-space signature classifier | Routes each scan to its validated model. Safely rejects unsupported scans (e.g. dental X-rays). |
| **3. Temperature Scaling** | $p_i = \text{softmax}(z_i / T)$ optimized on validation NLL | Eliminates uncalibrated overconfidence and produces true empirical confidence. |
| **4. MC Dropout Uncertainty** | 10 stochastic forward passes with test-time active dropout | Quantifies epistemic (model) uncertainty through variance and Shannon entropy. |
| **5. Penultimate OOD Detection** | Free-Energy $E(x) = -T \cdot \log \sum \exp(z_i / T)$ + prototype cosine distance | Flags unfamiliar or non-standard anatomic scans as `HIGH RISK (OOD)`. |

---

## 4. Hackathon Presentation: Key Demonstration Scenarios

### Case A — Reliable Concordant Prediction
- **Input**: Clear frontal Chest X-ray (`case_a_reliable_cxr.png`)
- **Router**: Lung ResNet-18 selected
- **Trust Audit**: Calibrated Confidence = High, Uncertainty = LOW, OOD = LOW RISK, Stability = 100%
- **Verdict**: **`AI RELIABILITY: HIGH`** &rarr; Standard AI-assisted pulmonology review recommended.

### Case B — Misleading High Softmax Confidence (Judges' Demo)
- **Input**: Unfamiliar / OOD textured pattern (`case_b_misleading_ood.png`)
- **Traditional AI**: Predicts 95%+ probability of disease.
- **TrustMed Audit**: Uncertainty = **HIGH**, OOD = **HIGH RISK**, Stability = **LOW**
- **Verdict**: **`AI RELIABILITY: LOW`** &rarr; **⚠️ CRITICAL TRUST WARNING: Result not dependable. Immediate human specialist review required.**

### Case C — Unsupported Modality Safe Rejection
- **Input**: Dental Panoramic X-ray (`case_c_dental_unsupported.png`)
- **Router**: Detected Dental X-ray / Unsupported modality.
- **Verdict**: **SAFE REJECTION** &rarr; *"Unsupported image type. No validated TrustMed model is available. Diagnostic inference refused."*

### Case D — Brain Tumor Segmentation
- **Input**: BraTS 2021 Brain MRI Axial Slice (`case_d_brain_mri.png`)
- **Model**: Lightweight 2D U-Net
- **Output**: Binary tumor mask, pixel coverage area, and translucent cyan/yellow lesion overlay.

---

## 5. Real-Time CPU Benchmark Results

Benchmarks measured on **Intel Core i5-1334U (12 GB RAM, Integrated Graphics)**:

* **Model Startup Load Time**: `634.5 ms` (Loaded once as a singleton)
* **Single Scan Pipeline Latency**: `321.9 ms` total
  * Quality Gate: `0.7 ms`
  * Specialty Router: `0.8 ms`
  * ResNet-18 Forward Pass: `103.8 ms`
  * Trust Engine (MC Dropout 10 passes + OOD + Stability + Grad-CAM): `214.8 ms`
* **Batch Throughput**:
  * 1 Image: `333 ms` (`3.0 scans/s`)
  * 2 Images: `658 ms` (`3.0 scans/s`)
  * 3 Images: `988 ms` (`3.0 scans/s`)
  * 4 Images: `1299 ms` (`3.1 scans/s`)
* **RAM Footprint**: `590.5 MB`

---

## 6. Model Evaluation Metrics

All metrics computed from patient-stratified test splits:

| Specialty | Modality | Architecture | Primary Metric | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Lung** | Chest X-ray | ResNet-18 | Macro-AUROC: **0.850** | 0.824 | 0.871 |
| **Brain** | Brain MRI | Standard U-Net | Dice: **0.820** / IoU: **0.730** | 0.852 | 0.941 |
| **Skin** | Dermoscopy | ResNet-18 | Macro-AUROC: **0.880** | 0.835 | 0.892 |
| **Retina** | Fundus Photo | ResNet-18 | Macro-AUROC: **0.880** | 0.810 | 0.885 |
| **Ablation** | CXR + CBAM Attention | ResNet-18 + CBAM | Macro-AUROC: **0.865** (+0.015) | 0.839 | 0.880 |

---

## 7. Quick Start & Execution

### 1. Requirements
Ensure Python 3.10+ and Node.js 18+ are available.
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 3. Open Healthcare Dashboard
Navigate to:
```text
http://127.0.0.1:8000/
```
*(The FastAPI backend automatically serves the pre-bundled React + Tailwind frontend dashboard and demo assets.)*

Alternatively, run the Vite development server:
```bash
cd frontend
npm run dev
```

### 4. Run Automated Test Suite
```bash
pytest tests/test_all.py -v
```
*(15 automated tests covering Quality Gate, Router, MC Dropout, OOD, Stability, Models, Batching, and PDF Export).*

---

## 8. Safety & Ethical Compliance

1. **Not a Medical Device**: TrustMed is an experimental AI decision-support prototype.
2. **Zero Treatment or Prescription Generation**: The system never provides drug prescriptions or autonomous treatment instructions.
3. **Transparent Escalation**: Unreliable predictions automatically escalate to `HIGH PRIORITY HUMAN REVIEW`.
4. **Local Execution**: All inference executes locally on CPU, ensuring zero external transmission of patient data.
