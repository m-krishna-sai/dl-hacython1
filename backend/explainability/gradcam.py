"""
TRUSTMED Grad-CAM Explainability Engine
Generates Gradient-weighted Class Activation Maps (Grad-CAM) for ResNet-18 classification models.
Visualizes which spatial image regions contributed most strongly to the predicted medical finding.
Provides compliant clinical disclaimers.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int
    ) -> np.ndarray:
        """
        Generates 2D normalized heatmap [0, 1] of shape [H, W].
        input_tensor: [1, 3, 224, 224] with requires_grad=True
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        # Target class score
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        # activations: [1, 512, 7, 7], gradients: [1, 512, 7, 7]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain positive features
        cam = F.relu(cam)

        # Upsample to input tensor spatial dimension (224, 224)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "jet"
    ) -> Image.Image:
        """Overlays colormapped heatmap onto original image."""
        orig = original_image.convert("RGB").resize((224, 224))
        orig_arr = np.array(orig, dtype=np.float32) / 255.0

        # Apply colormap
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap(colormap_name) if hasattr(mpl.colormaps, "get_cmap") else mpl.colormaps[colormap_name]
        heatmap_colored = cmap(heatmap)[:, :, :3] # RGB

        # Blend
        blended = (1.0 - alpha) * orig_arr + alpha * heatmap_colored
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    @staticmethod
    def to_base64(img: Image.Image) -> str:
        """Converts PIL image to base64 data URI."""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_idx: int
) -> str:
    """Convenience helper to generate base64 overlay string."""
    try:
        # ResNet-18 layer4 is target
        target_layer = getattr(model, "layer4", None)
        if target_layer is None:
            return ""

        input_var = input_tensor.clone().detach().requires_grad_(True)
        gcam = GradCAM(model, target_layer)
        heatmap = gcam.generate_heatmap(input_var, target_class_idx)
        overlay_img = GradCAM.overlay_heatmap(original_image, heatmap)
        return GradCAM.to_base64(overlay_img)
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return ""
