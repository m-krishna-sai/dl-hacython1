"""
End-to-end FastAPI test using TestClient
Tests both /api/validate and /api/analyze for all 4 specialties
"""
from fastapi.testclient import TestClient
from backend.main import app
import os

client = TestClient(app)

cases = [
    ("test_images/1_lung/normal.jpg", "lung", "Normal"),
    ("test_images/1_lung/pneumonia.jpg", "lung", "Effusion"),
    ("test_images/2_brain/normal.jpg", "brain", "Normal"),
    ("test_images/2_brain/tumor.jpg", "brain", "Brain Tumor"),
    ("test_images/3_skin/normal_benign_nevus.jpg", "skin", "Melanocytic Nevus (NV) (Benign)"),
    ("test_images/3_skin/malignant_melanoma.jpg", "skin", "Melanoma (MEL) (Malignant)"),
    ("test_images/4_retina/healthy_normal.jpg", "retina", "Normal (0 - No DR)"),
    ("test_images/4_retina/diabetic_retinopathy.jpg", "retina", "Diabetic Retinopathy (2 - Moderate)"),
]

print("=== Testing FastAPI Endpoints ===")
with client:
    for path, specialty, expected_keyword in cases:
        fname = os.path.basename(path)
        with open(path, "rb") as f:
            file_bytes = f.read()

        # 1. Test /api/validate
        val_resp = client.post(
            "/api/validate",
            files={"file": (fname, file_bytes, "image/jpeg")},
            data={"specialty": specialty}
        )
        val_data = val_resp.json()
        valid = val_data.get("valid", False)

        # 2. Test /api/analyze
        ana_resp = client.post(
            "/api/analyze",
            files=[("files", (fname, file_bytes, "image/jpeg"))],
            data={"specialty": specialty}
        )
        ana_data = ana_resp.json()
        results = ana_data.get("results", [])
        if results:
            res0 = results[0]
            pred = res0.get("predicted_class", "N/A")
            conf = round(res0.get("confidence", 0) * 100, 1)
            is_normal = res0.get("is_normal", None)
            disclaimer = bool(res0.get("medical_disclaimer"))
            print(f"[{specialty.upper():6s}] Validate: {valid} | Pred: {pred:35s} ({conf}%) | Normal: {is_normal} | Disclaimer: {disclaimer}")
        else:
            print(f"[{specialty.upper():6s}] Validate: {valid} | Analyze ERROR: {ana_resp.text}")

print("=== All FastAPI Endpoint Tests Completed ===")
