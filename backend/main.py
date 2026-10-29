"""
TRUSTMED FastAPI Main Application
Provides RESTful APIs for:
- Multi-specialty medical image analysis (1-4 images)
- Trust Engine evaluation (calibration, uncertainty, OOD, stability, quality)
- Clinical PDF report generation
- Model performance metrics & dataset inspection reports
- Startup singleton model preloading
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.models.loader import get_registry
from backend.api.inference import process_single_image, process_batch
from backend.api.reports import generate_pdf_report
from backend.api.health import get_system_health
from backend.trust.quality import quality_gate
from backend.routing.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload all specialty models once at startup
    print("[TRUSTMED] Starting FastAPI application server...")
    registry = get_registry()
    registry.load_all_models()
    yield
    print("[TRUSTMED] Shutting down application server...")

app = FastAPI(
    title="TrustMed — Self-Aware Multi-Specialty Medical AI API",
    description="Decision-support AI with confidence calibration, uncertainty estimation, and OOD detection.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.get("/api/health")
def health():
    """Returns system status, model cache status, and hardware resource metrics."""
    return get_system_health()

@app.get("/models")
@app.get("/api/models")
def get_models_info():
    """Returns list of registered models, tasks, architectures, and weights."""
    registry = get_registry()
    return {
        "status": "ready",
        "models": {
            "lung": {"architecture": "ResNet-18", "task": "Multi-label classification (14 classes)", "pretrained": "ImageNet", "classes": 14},
            "brain": {"architecture": "U-Net", "task": "Brain tumor segmentation", "loss": "Dice + BCE", "channels": 3},
            "skin": {"architecture": "ResNet-18", "task": "Dermoscopy classification (8 classes)", "pretrained": "ImageNet", "classes": 8},
            "retina": {"architecture": "ResNet-18", "task": "Diabetic retinopathy grading (5 grades)", "pretrained": "ImageNet", "classes": 5}
        }
    }

@app.get("/datasets")
@app.get("/api/datasets")
def get_datasets_report():
    """Returns dataset inspection QA metrics, distributions, and licensing data."""
    report_file = WORKSPACE_ROOT / "reports" / "dataset_report.json"
    if not report_file.exists():
        from training.inspection import run_inspection
        run_inspection(WORKSPACE_ROOT)

    with open(report_file, "r") as f:
        data = json.load(f)
    return data

@app.get("/benchmark")
@app.get("/api/benchmark")
@app.get("/api/performance")
def get_model_performance():
    """Returns real benchmark metrics, confusion matrices, calibration curves, and ablation data."""
    perf_file = WORKSPACE_ROOT / "reports" / "model_performance.json"
    if not perf_file.exists():
        from training.train_all import main as run_train
        run_train()

    with open(perf_file, "r") as f:
        data = json.load(f)
    return data

class Base64ImageItem(BaseModel):
    name: str
    data: str # base64 data URI
    specialty: Optional[str] = "auto"

class AnalyzeBase64Request(BaseModel):
    images: List[Base64ImageItem]

@app.post("/api/analyze/base64")
async def analyze_images_base64(payload: AnalyzeBase64Request):
    """
    Analyzes 1 to 4 images provided as base64 data URIs.
    """
    if not payload.images:
        raise HTTPException(status_code=400, detail="At least 1 image is required.")
    if len(payload.images) > 4:
        raise HTTPException(status_code=400, detail="Maximum of 4 images allowed per batch.")

    img_tuples = []
    for item in payload.images:
        try:
            b64_str = item.data
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tuples.append((pil_img, item.name))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format for {item.name}: {str(e)}")

    return process_batch(img_tuples)

@app.post("/predict/batch")
@app.post("/api/analyze")
async def analyze_images_form(
    files: List[UploadFile] = File(...),
    specialty: Optional[str] = Form("auto")
):
    """
    Analyzes 1 to 4 uploaded multipart image files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least 1 file is required.")
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 images allowed per batch.")

    img_tuples = []
    for f in files:
        try:
            contents = await f.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_tuples.append((pil_img, f.filename))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image {f.filename}: {str(e)}")

    return process_batch(img_tuples, requested_specialty=specialty)

@app.post("/api/validate")
@app.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    specialty: Optional[str] = Form("auto")
):
    """
    Fast validation gate checking file integrity, image quality, and specialty suitability
    without executing heavy neural network forward passes.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {
            "valid": False,
            "error_type": "corrupt",
            "title": "Could not read image file",
            "message": f"The uploaded file is corrupt or in an unreadable format: {str(e)}",
            "filename": file.filename
        }

    # 1. Quality Check
    q_res = quality_gate.evaluate(pil_img)
    if not q_res.get("passed", False):
        return {
            "valid": False,
            "error_type": "quality",
            "title": "Image quality is too low",
            "message": "Please upload a clearer medical image so TRUSTMED can analyze it reliably.",
            "reasons": q_res.get("reasons", []),
            "quality": q_res,
            "filename": file.filename
        }

    # 2. Modality & Specialty Routing
    r_res = router.route(pil_img, requested_specialty=specialty)
    if not r_res.get("supported", False):
        return {
            "valid": False,
            "error_type": "unsupported",
            "title": "Sorry, this image isn't supported.",
            "message": "TRUSTMED currently supports:\n• Lung — Chest X-rays\n• Brain — Brain MRI\n• Skin — Dermoscopic skin lesion images\n• Retina — Retinal fundus images",
            "routing": r_res,
            "filename": file.filename
        }

    # 3. Specialty Mismatch Check
    if specialty and specialty != "auto" and r_res.get("specialty") != specialty:
        modality_name = r_res.get("modality", r_res.get("specialty", "different modality"))
        return {
            "valid": False,
            "error_type": "mismatch",
            "title": "Image type doesn't match the selected specialty.",
            "message": f"This image appears to be a {modality_name}, but '{specialty.capitalize()}' was selected.",
            "detected_specialty": r_res.get("specialty"),
            "detected_modality": r_res.get("modality"),
            "selected_specialty": specialty,
            "routing": r_res,
            "filename": file.filename
        }

    modality_name = r_res.get("modality", r_res.get("specialty", "Medical scan"))
    return {
        "valid": True,
        "error_type": None,
        "title": "Image looks suitable for analysis ✓",
        "message": f"Successfully verified as a {modality_name}. Quality check passed.",
        "detected_specialty": r_res.get("specialty"),
        "detected_modality": r_res.get("modality"),
        "quality": q_res,
        "routing": r_res,
        "filename": file.filename
    }

@app.post("/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    specialty: Optional[str] = Form("auto")
):
    """
    Analyzes a single uploaded medical image file.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        res = process_single_image(pil_img, file.filename, requested_specialty=specialty)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image {file.filename}: {str(e)}")

@app.post("/api/report/pdf")
async def download_pdf_report(analysis_result: dict):
    """
    Generates and downloads a clinical PDF report for an analyzed image.
    Safety gate: Refuses report generation for invalid or rejected images.
    """
    if analysis_result.get("status", "").startswith("REJECTED") or analysis_result.get("valid") is False:
        raise HTTPException(status_code=400, detail="Cannot generate report for an invalid or rejected image.")

    try:
        pdf_bytes = generate_pdf_report(analysis_result)
        filename = f"TrustMed_Report_{analysis_result.get('filename', 'scan').split('.')[0]}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount demo cases directory
demo_dir = WORKSPACE_ROOT / "data" / "demo_cases"
if demo_dir.exists():
    app.mount("/data/demo_cases", StaticFiles(directory=str(demo_dir)), name="demo_cases")

# Mount frontend dist if built
dist_dir = WORKSPACE_ROOT / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(dist_dir / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
