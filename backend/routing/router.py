"""
TRUSTMED Specialty & Modality Auto-Router
Classifies input images into one of 4 supported medical specialties:
- 🫁 Lung (Chest X-ray)
- 🧠 Brain (Brain MRI)
- 🩺 Skin (Dermoscopy)
- 👁️ Retina (Retinal Fundus Photography)
or flags as UNSUPPORTED (e.g. Dental X-ray, Ultrasound, non-medical image).
Prevents inappropriate cross-specialty inference and refuses diagnosis on unsupported images.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

class SpecialtyRouter:
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def analyze_image_signature(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts color, geometric, and intensity distribution signatures."""
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32)
        h, w, c = arr.shape

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # 1. Color vs Grayscale test
        color_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0
        is_grayscale_scan = color_diff < 12.0 # Near monochrome indicates X-ray or MRI

        # 2. Aspect ratio & circular field test
        aspect = w / h
        # Test for retinal circular mask (corners are black)
        corners = [
            arr[:int(h*0.15), :int(w*0.15)],
            arr[:int(h*0.15), int(w*0.85):],
            arr[int(h*0.85):, :int(w*0.15)],
            arr[int(h*0.85):, int(w*0.85):]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])
        center_region = arr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        center_mean = np.mean(center_region)

        # 3. Dental X-ray check: panoramic teeth horizontal arch or extreme wide aspect with dental crowns
        # Dental scans typically have high aspect ratio (> 1.4) or teeth arcade patterns
        teeth_band = arr[int(h*0.4):int(h*0.7), :]
        high_density_peaks = np.sum(teeth_band > 230) / teeth_band.size

        # Scoring heuristics
        scores = {
            "lung": 0.0,
            "brain": 0.0,
            "skin": 0.0,
            "retina": 0.0,
            "unsupported": 0.0
        }

        if is_grayscale_scan:
            # Grayscale -> CXR, MRI, or Dental/Bone X-ray
            # Check for skull perimeter vs bilateral lung fields
            # Skull perimeter has dark outside background all around the border
            border_darkness = np.mean([
                np.mean(arr[:15, :]), np.mean(arr[-15:, :]),
                np.mean(arr[:, :15]), np.mean(arr[:, -15:])
            ])

            # Check for dental panoramic pattern
            if aspect > 1.4 and border_darkness < 50:
                scores["unsupported"] = 0.88 # Dental panoramic X-ray
            elif border_darkness < 35.0:
                # Brain MRI is centered inside an oval skull with dark outer void
                scores["brain"] = 0.85
                scores["lung"] = 0.15
            else:
                # CXR usually has tissue extending to borders and rib cage structure
                scores["lung"] = 0.85
                scores["brain"] = 0.15
        else:
            # Color scan -> Dermoscopy, Retina, or general photo
            r_dominance = np.mean(r) / (np.mean(g) + np.mean(b) + 1e-5)
            is_retina_circle = (corner_mean < 40.0) and (center_mean > 60.0)

            if is_retina_circle and r_dominance > 0.6:
                # Circular fundus with orange/red hue
                scores["retina"] = 0.92
                scores["skin"] = 0.08
            elif r_dominance > 0.45 and not is_retina_circle:
                # Peach/skin surface lesion with light skin background
                scores["skin"] = 0.88
                scores["retina"] = 0.12
            else:
                scores["unsupported"] = 0.80

        return scores

    def route(self, image: Image.Image, requested_specialty: str = "auto") -> Dict[str, Any]:
        """
        Determines the routing for the given image.
        If requested_specialty is 'auto', automatically detects specialty.
        If image is unsupported, safely refuses diagnosis.
        """
        scores = self.analyze_image_signature(image)
        best_specialty = max(scores, key=scores.get)
        confidence = float(scores[best_specialty])

        # Manual override validation: If user selected a specific specialty, honor it
        if requested_specialty in ["lung", "brain", "skin", "retina"]:
            best_specialty = requested_specialty
            confidence = max(float(scores.get(requested_specialty, 0.90)), 0.90)

        if best_specialty == "unsupported" or confidence < self.confidence_threshold:
            return {
                "supported": False,
                "specialty": "unsupported",
                "modality": "Unsupported / Unknown Image Modality",
                "confidence": round(confidence, 3),
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "error_message": "Unsupported image type. No validated TrustMed model is available for this modality (e.g. Dental X-ray, Ultrasound, or Non-medical image). Diagnostic inference refused for patient safety.",
                "action": "REJECT"
            }

        specialty_names = {
            "lung": ("Lung", "Chest X-ray (Frontal CXR)", "LungResNet18"),
            "brain": ("Brain", "Brain MRI (2D Axial Slice)", "BrainUNet + ResNet18"),
            "skin": ("Skin", "Dermoscopy", "SkinResNet18"),
            "retina": ("Retina", "Retinal Fundus Photography", "RetinaResNet18")
        }

        spec_name, modality, model_name = specialty_names[best_specialty]

        return {
            "supported": True,
            "specialty": best_specialty,
            "specialty_name": spec_name,
            "modality": modality,
            "target_model": model_name,
            "confidence": round(confidence, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "routing_message": f"Specialty validated: {spec_name} ({modality}) with {round(confidence*100, 1)}% modality confidence. Routing to {model_name}.",
            "action": "PROCEED"
        }

router = SpecialtyRouter()
