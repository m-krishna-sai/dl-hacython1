"""
TRUSTMED Hackathon Judge Demonstration Cases Generator
Generates curated test cases for presentation:
- Case A: Reliable Prediction (Clear Chest X-ray, High Reliability)
- Case B: Misleading High Confidence (OOD / Unfamiliar image with high nominal probability but high uncertainty & OOD -> LOW Reliability)
- Case C: Unsupported Modality (Dental panoramic X-ray -> Safe Rejection)
- Case D: Brain MRI (Tumor segmentation with U-Net overlay)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = WORKSPACE_ROOT / "data" / "demo_cases"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

def generate_case_a_reliable_lung():
    """Case A: High quality, in-distribution Chest X-ray showing consolidation."""
    arr = np.ones((224, 224), dtype=np.uint8) * 35
    arr[:, 95:130] = 160 # Spine / Mediastinum
    # Bilateral lung fields
    y, x = np.ogrid[:224, :224]
    mask_l = ((x - 55)**2 / (35**2) + (y - 110)**2 / (65**2)) <= 1
    mask_r = ((x - 170)**2 / (35**2) + (y - 110)**2 / (65**2)) <= 1
    arr[mask_l] = 45
    arr[mask_r] = 45
    for r in range(40, 190, 25):
        arr[r:r+3, 30:200] = np.clip(arr[r:r+3, 30:200] + 40, 0, 255)
    # Dense focal infiltrate in right lower lobe
    arr[135:175, 145:190] += 70
    img = Image.fromarray(arr).convert("RGB")
    img.save(DEMO_DIR / "case_a_reliable_cxr.png")
    print(f"Generated Case A: {DEMO_DIR / 'case_a_reliable_cxr.png'}")

def generate_case_b_misleading_ood():
    """Case B: Unfamiliar / OOD image that causes high raw probability but high uncertainty & OOD."""
    # Synthetic alien-texture / noise-pattern resembling X-ray contrast but unnatural frequencies
    np.random.seed(999)
    noise = np.random.normal(loc=120, scale=40, size=(224, 224)).astype(np.uint8)
    y, x = np.ogrid[:224, :224]
    rings = (np.sin(np.sqrt((x-112)**2 + (y-112)**2) / 6.0) * 127 + 128).astype(np.uint8)
    hybrid = ((noise.astype(float) * 0.4) + (rings.astype(float) * 0.6)).astype(np.uint8)
    img = Image.fromarray(hybrid).convert("RGB")
    img.save(DEMO_DIR / "case_b_misleading_ood.png")
    print(f"Generated Case B: {DEMO_DIR / 'case_b_misleading_ood.png'}")

def generate_case_c_unsupported_dental():
    """Case C: Unsupported Dental Panoramic X-ray."""
    arr = np.zeros((160, 280), dtype=np.uint8)
    arr[30:130, 20:260] = 130
    arr[65:90, 30:250] = 235 # Dense dental arch with crowns and fillings
    # Metal artifacts radiating
    for c in range(50, 230, 30):
        arr[50:110, c:c+2] = 255
    img = Image.fromarray(arr).convert("RGB")
    img.save(DEMO_DIR / "case_c_dental_unsupported.png")
    print(f"Generated Case C: {DEMO_DIR / 'case_c_dental_unsupported.png'}")

def generate_case_d_brain_mri():
    """Case D: Brain MRI axial slice with clear tumor lesion for U-Net."""
    y, x = np.ogrid[:224, :224]
    skull = ((x - 112)**2 / 85**2 + (y - 112)**2 / 95**2) <= 1
    arr = np.zeros((224, 224), dtype=np.uint8)
    arr[skull] = 130
    # Ventricles
    vent_l = ((x - 100)**2 / (8**2) + (y - 110)**2 / (25**2)) <= 1
    vent_r = ((x - 124)**2 / (8**2) + (y - 110)**2 / (25**2)) <= 1
    arr[vent_l] = 20
    arr[vent_r] = 20
    # Tumor
    t_mask = ((x - 145)**2 / (18**2) + (y - 95)**2 / (22**2)) <= 1
    arr[t_mask] = 220
    img = Image.fromarray(arr).convert("RGB")
    img.save(DEMO_DIR / "case_d_brain_mri.png")
    print(f"Generated Case D: {DEMO_DIR / 'case_d_brain_mri.png'}")

if __name__ == "__main__":
    generate_case_a_reliable_lung()
    generate_case_b_misleading_ood()
    generate_case_c_unsupported_dental()
    generate_case_d_brain_mri()
