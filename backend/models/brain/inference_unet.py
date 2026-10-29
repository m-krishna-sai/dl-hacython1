"""
TRUSTMED Brain U-Net Inference & Segmentation Visualizer
Generates binary tumor masks and translucent RGB overlays for Brain MRI slices.
Calculates tumor pixel area and volume coverage.
"""

import torch
import numpy as np
from PIL import Image
import io
import base64

def run_brain_segmentation(
    unet_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    threshold: float = 0.5
) -> dict:
    """
    input_tensor: [1, 3, 224, 224]
    original_image: PIL Image
    """
    unet_model.eval()
    with torch.no_grad():
        logits = unet_model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probs > threshold).astype(np.uint8)

    # Calculate lesion statistics
    total_pixels = pred_mask.size
    tumor_pixels = int(np.sum(pred_mask))
    coverage_pct = round((tumor_pixels / total_pixels) * 100.0, 2)
    has_tumor = tumor_pixels > 50

    # Create visual overlays
    orig_resized = original_image.convert("RGB").resize((224, 224))
    orig_np = np.array(orig_resized, dtype=np.uint8)

    # 1. Standalone binary mask image (white on black)
    mask_img = Image.fromarray(pred_mask * 255).convert("L")

    # 2. Translucent colored overlay (cyan/teal lesion overlay with yellow border)
    overlay_np = orig_np.copy().astype(np.float32)
    if has_tumor:
        # Tint tumor region with cyan [0, 220, 220]
        alpha = 0.50
        cyan_color = np.array([0, 220, 220], dtype=np.float32)
        tumor_idx = pred_mask == 1
        overlay_np[tumor_idx] = (1.0 - alpha) * overlay_np[tumor_idx] + alpha * cyan_color

        # Find perimeter/contour for distinct border
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(pred_mask, iterations=1)
        border = np.logical_xor(dilated, pred_mask)
        overlay_np[border] = [255, 230, 0] # Bright yellow outline

    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return {
        "has_tumor": bool(has_tumor),
        "tumor_pixels": tumor_pixels,
        "coverage_percentage": coverage_pct,
        "mask_base64": to_b64(mask_img),
        "overlay_base64": to_b64(overlay_img)
    }
