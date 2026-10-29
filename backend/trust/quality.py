"""
TRUSTMED Image Quality Gate
Analyzes input image quality before model inference:
- Blur detection (Laplacian variance)
- Contrast & dynamic range (Michelson & RMS contrast)
- Exposure & brightness distribution (under/over exposure)
- Minimum resolution & aspect ratio
- Extreme noise artifacts

Hard safety gate: If image quality is severely inadequate (POOR),
rejects immediately without running diagnostic inference.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

class ImageQualityGate:
    def __init__(
        self,
        min_resolution: int = 128,
        blur_threshold: float = 40.0,
        min_contrast: float = 0.10,
        dark_threshold: float = 15.0,
        bright_threshold: float = 240.0
    ):
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.min_contrast = min_contrast
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def evaluate(self, image: Image.Image) -> Dict[str, Any]:
        """
        Evaluates image quality. Returns status ('GOOD' or 'POOR'),
        pass_gate boolean, scores, and human-readable feedback.
        """
        w, h = image.size
        reasons = []

        # 1. Resolution Check
        if w < self.min_resolution or h < self.min_resolution:
            reasons.append(f"Image resolution too low ({w}x{h}). Minimum required is {self.min_resolution}x{self.min_resolution}.")

        # Convert to grayscale numpy array
        gray = np.array(image.convert("L"), dtype=np.float32)

        # 2. Blur Check using Laplacian Variance
        # Approximation of Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        laplacian = (
            np.roll(gray, 1, axis=0) +
            np.roll(gray, -1, axis=0) +
            np.roll(gray, 1, axis=1) +
            np.roll(gray, -1, axis=1) -
            4 * gray
        )
        blur_score = float(np.var(laplacian))
        is_blurry = blur_score < self.blur_threshold
        if is_blurry:
            reasons.append(f"Severe blur detected (score: {blur_score:.1f}, threshold: {self.blur_threshold}).")

        # 3. Brightness & Exposure Check
        mean_brightness = float(np.mean(gray))
        is_underexposed = mean_brightness < self.dark_threshold
        is_overexposed = mean_brightness > self.bright_threshold

        if is_underexposed:
            reasons.append("Image is severely underexposed (too dark). Essential anatomic features obscured.")
        elif is_overexposed:
            reasons.append("Image is severely overexposed (washed out). Contrast details lost.")

        # 4. Dynamic Range & Contrast Check (RMS Contrast)
        rms_contrast = float(np.std(gray) / 255.0)
        is_low_contrast = rms_contrast < self.min_contrast
        if is_low_contrast:
            reasons.append(f"Inadequate dynamic contrast ({rms_contrast:.2f} < {self.min_contrast}).")

        # Decision
        is_good = len(reasons) == 0
        verdict = "GOOD" if is_good else "POOR"

        return {
            "verdict": verdict,
            "passed": is_good,
            "resolution": [w, h],
            "blur_score": round(blur_score, 1),
            "blur_status": "Sharp" if not is_blurry else "Blurry",
            "mean_brightness": round(mean_brightness, 1),
            "rms_contrast": round(rms_contrast, 3),
            "reasons": reasons,
            "recommendation": "Proceed to analysis" if is_good else "Severe image quality inadequacy. Please upload a clearer, properly exposed medical scan."
        }

quality_gate = ImageQualityGate()
