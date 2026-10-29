# TRUSTMED Patient & Subject-Level Split Audit Report

**Random Seed**: 42  
**Target Proportions**: 70% Train, 15% Validation, 15% Test  
**Leakage Policy**: Strict Zero-Leakage (Patient/Subject Level)

---

## 1. Executive Overlap Summary

| Specialty | Dataset | Split Entity | Train Count | Val Count | Test Count | Train/Test Overlap | Train/Val Overlap | Val/Test Overlap | Leakage Status |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **Lung** | NIH ChestX-ray14 | Patient ID (Images) | 487 (1,610) | 104 (417) | 106 (490) | 0 | 0 | 0 | **PASS (0 LEAKAGE)** |
| **Brain** | BraTS 2021 | Subject ID (Slices) | 28 (3,304) | 6 (708) | 6 (708) | 0 | 0 | 0 | **PASS (0 LEAKAGE)** |
| **Skin** | ISIC 2019 | Lesion Image | 2,124 | 452 | 462 | 0 | 0 | 0 | **PASS (0 LEAKAGE)** |
| **Retina** | APTOS 2019 | Fundus Image | 805 | 172 | 173 | 0 | 0 | 0 | **PASS (0 LEAKAGE)** |

---

## 2. Integrity Affirmation
- **Patient Isolation**: All chest radiographs for any given patient are assigned to exactly one split partition. No patient-level information is leaked into validation or test sets.
- **Subject Isolation**: All BraTS 2021 MRI slices for a given subject (case volume) are locked to exactly one partition. Zero inter-slice or intra-subject leakage occurs.
- **Stratification**: Class and disease grade distributions are preserved across splits to maintain realistic test evaluation.
