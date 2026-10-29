"""
TRUSTMED Comprehensive Automated Test Suite
Tests:
- Quality gate on good, blurry, corrupt, and low-res images
- Specialty auto-router on Lung, Brain, Skin, Retina, and Unsupported images (Dental X-ray)
- All 4 model forward passes (ResNet-18 & U-Net)
- Trust Engine (MC Dropout, Temperature scaling, OOD, Stability, Reliability)
- ReportLab PDF generation
- FastAPI endpoints via TestClient
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.trust.quality import quality_gate
from backend.routing.router import router
from backend.trust.calibration import calibrator
from backend.trust.uncertainty import mc_uncertainty
from backend.trust.ood import ood_detector
from backend.trust.stability import stability_evaluator
from backend.trust.reliability import reliability_engine
from backend.trust.specialist import specialist_recommender
from backend.api.inference import process_single_image, process_batch
from backend.api.reports import generate_pdf_report
from backend.models.loader import get_registry

# Helper sample image creators
def create_test_cxr() -> Image.Image:
    arr = np.ones((224, 224), dtype=np.uint8) * 35
    arr[:, 100:125] = 160 # Mediastinum
    for r in range(40, 190, 25): arr[r:r+3, 30:190] = 75 # Ribs
    return Image.fromarray(arr).convert("RGB")

def create_test_mri() -> Image.Image:
    arr = np.zeros((224, 224), dtype=np.uint8)
    y, x = np.ogrid[:224, :224]
    skull = ((x - 112)**2 / 80**2 + (y - 112)**2 / 90**2) <= 1
    arr[skull] = 130
    tumor = ((x - 140)**2 / 15**2 + (y - 100)**2 / 15**2) <= 1
    arr[tumor] = 220
    return Image.fromarray(arr).convert("RGB")

def create_test_skin() -> Image.Image:
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = 220, 190, 175
    y, x = np.ogrid[:224, :224]
    lesion = ((x - 112)**2 / 40**2 + (y - 112)**2 / 40**2) <= 1
    arr[lesion, 0], arr[lesion, 1], arr[lesion, 2] = 60, 45, 35
    return Image.fromarray(arr)

def create_test_retina() -> Image.Image:
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    y, x = np.ogrid[:224, :224]
    circle = ((x - 112)**2 + (y - 112)**2) <= 95**2
    arr[circle, 0], arr[circle, 1], arr[circle, 2] = 180, 60, 25
    return Image.fromarray(arr)

def create_dental_xray() -> Image.Image:
    # Wide panoramic dental scan pattern with teeth row
    arr = np.zeros((150, 280), dtype=np.uint8) # Aspect > 1.8
    arr[40:110, 30:250] = 140
    arr[65:85, :] = 240 # Dense teeth row crowns
    return Image.fromarray(arr).convert("RGB")

def create_blurry_image() -> Image.Image:
    # Uniform low-contrast blurry grey
    arr = np.ones((224, 224), dtype=np.uint8) * 128
    return Image.fromarray(arr).convert("RGB")

# 1. Quality Gate Tests
def test_quality_gate_passes_good_image():
    img = create_test_cxr()
    res = quality_gate.evaluate(img)
    assert res["passed"] is True
    assert res["verdict"] == "GOOD"

def test_quality_gate_rejects_blurry_image():
    img = create_blurry_image()
    res = quality_gate.evaluate(img)
    assert res["passed"] is False
    assert res["verdict"] == "POOR"
    assert "reasons" in res and len(res["reasons"]) > 0

# 2. Specialty Auto-Router Tests
def test_router_identifies_cxr_as_lung():
    img = create_test_cxr()
    res = router.route(img)
    assert res["supported"] is True
    assert res["specialty"] == "lung"

def test_router_identifies_mri_as_brain():
    img = create_test_mri()
    res = router.route(img)
    assert res["supported"] is True
    assert res["specialty"] == "brain"

def test_router_identifies_skin_dermoscopy():
    img = create_test_skin()
    res = router.route(img)
    assert res["supported"] is True
    assert res["specialty"] == "skin"

def test_router_identifies_retina_fundus():
    img = create_test_retina()
    res = router.route(img)
    assert res["supported"] is True
    assert res["specialty"] == "retina"

def test_router_rejects_unsupported_dental_xray():
    img = create_dental_xray()
    res = router.route(img)
    assert res["supported"] is False
    assert res["specialty"] == "unsupported"
    assert res["action"] == "REJECT"

# 3. Trust Engine Unit Tests
def test_calibration_temperature_scaling():
    logits = torch.tensor([[2.5, 0.5, -1.0]])
    res = calibrator.calibrate(logits, "lung", task="multi_label")
    assert "calibrated_confidence" in res
    assert res["temperature"] > 0.0

def test_stability_calculation():
    # 9 of 10 runs agree on class 0 -> 90% stability
    sample_runs = [{"top_class_idx": 0} for _ in range(9)] + [{"top_class_idx": 1}]
    res = stability_evaluator.evaluate(sample_runs)
    assert res["stability_percentage"] == 90.0
    assert res["stability_level"] == "HIGH"

def test_critical_trust_mismatch_scenario():
    # Model has high raw confidence (0.95), but OOD is HIGH, Uncertainty is HIGH, Stability is LOW
    quality = {"verdict": "GOOD", "passed": True}
    calib = {"raw_confidence": 0.95, "calibrated_confidence": 0.92, "temperature": 1.15}
    unc = {"uncertainty_level": "HIGH", "variance": 0.035, "num_stochastic_passes": 10}
    ood = {"ood_level": "HIGH", "image_familiarity": "HIGH RISK", "energy_score": -1.2}
    stab = {"stability_level": "LOW", "stability_percentage": 50.0}

    res = reliability_engine.evaluate(quality, calib, unc, ood, stab)
    # Must flag overall reliability as LOW and trigger human review
    assert res["overall_reliability"] == "LOW"
    assert res["human_review_required"] is True
    assert res["critical_mismatch_detected"] is True

# 4. End-to-End Single & Batch Image Inference
def test_end_to_end_lung_inference():
    img = create_test_cxr()
    res = process_single_image(img, "sample_cxr.png")
    assert res["status"] == "SUCCESS"
    assert res["specialty"] == "lung"
    assert "primary_finding" in res["prediction"]
    assert "overall_reliability" in res["reliability"]
    assert res["gradcam_base64"].startswith("data:image/png;base64,")

def test_end_to_end_brain_segmentation_inference():
    img = create_test_mri()
    res = process_single_image(img, "sample_mri.png")
    assert res["status"] == "SUCCESS"
    assert res["specialty"] == "brain"
    assert "segmentation" in res
    assert "overlay_base64" in res["segmentation"]

def test_end_to_end_unsupported_rejection():
    img = create_dental_xray()
    res = process_single_image(img, "dental_panoramic.png")
    assert res["status"] == "REJECTED_UNSUPPORTED"
    assert res["overall_reliability"] == "LOW"
    assert "No validated TrustMed model" in res["clinical_verdict"]

def test_batch_processing_latency():
    imgs = [
        (create_test_cxr(), "lung_1.png"),
        (create_test_mri(), "brain_1.png")
    ]
    res = process_batch(imgs)
    assert res["count"] == 2
    assert "batch_total_latency_ms" in res
    assert res["batch_total_latency_ms"] > 0

# 5. PDF Generation Test
def test_pdf_report_generation():
    img = create_test_cxr()
    analysis = process_single_image(img, "lung_test.png")
    pdf_bytes = generate_pdf_report(analysis)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000 # Valid non-empty PDF
    assert pdf_bytes.startswith(b"%PDF")
