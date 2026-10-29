"""
TRUSTMED Singleton Model Loader & Registry
Loads trained PyTorch models once at startup into memory (CPU-friendly).
Maintains reference prototypes for OOD detection and temperature scaling parameters.
"""

import sys
from pathlib import Path
import torch
from typing import Dict, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from models.lung.resnet18_lung import LungResNet18, LUNG_CLASSES
from models.brain.unet_brain import BrainUNet
from models.brain.resnet18_brain import BrainResNet18
from models.skin.resnet18_skin import SkinResNet18, SKIN_CLASSES
from models.retina.resnet18_retina import RetinaResNet18, RETINA_CLASSES

class ModelRegistry:
    _instance = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.temperatures: Dict[str, float] = {}
        self.is_loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def load_all_models(self, models_dir: Optional[Path] = None):
        """Loads all specialty models into memory once at application startup."""
        if self.is_loaded:
            return

        if models_dir is None:
            models_dir = WORKSPACE_ROOT / "models"

        print(f"[TRUSTMED Registry] Loading all models onto {self.device}...")

        # 1. Lung Model
        lung_path = models_dir / "lung" / "best_resnet18.pth"
        lung_model = LungResNet18(num_classes=14, pretrained=False).to(self.device)
        temp_lung = 1.15
        if lung_path.exists():
            ckpt = torch.load(lung_path, map_location=self.device, weights_only=False)
            lung_model.load_state_dict(ckpt["model_state_dict"])
            temp_lung = ckpt.get("temperature", 1.15)
            proto = ckpt.get("prototypes", {}).get("centroid") if isinstance(ckpt.get("prototypes"), dict) else ckpt.get("feature_prototype_mean")
            if proto is not None:
                self.prototypes["lung"] = torch.tensor(proto, dtype=torch.float32)
        lung_model.eval()
        self.models["lung"] = lung_model
        self.temperatures["lung"] = temp_lung
        self.metadata["lung"] = {"classes": LUNG_CLASSES, "task": "multi_label"}

        # 2. Brain U-Net & Classifier
        brain_unet_path = models_dir / "brain" / "best_unet.pth"
        brain_unet = BrainUNet(in_channels=3, out_channels=1, base_features=32).to(self.device)
        if brain_unet_path.exists():
            ckpt = torch.load(brain_unet_path, map_location=self.device, weights_only=False)
            brain_unet.load_state_dict(ckpt["model_state_dict"])
        brain_unet.eval()
        self.models["brain_unet"] = brain_unet

        brain_cls_path = models_dir / "brain" / "best_resnet18.pth"
        brain_cls = BrainResNet18(num_classes=2, pretrained=False).to(self.device)
        temp_brain = 1.10
        if brain_cls_path.exists():
            ckpt = torch.load(brain_cls_path, map_location=self.device, weights_only=False)
            brain_cls.load_state_dict(ckpt["model_state_dict"])
            temp_brain = ckpt.get("temperature", 1.10)
        brain_cls.eval()
        self.models["brain_cls"] = brain_cls
        self.temperatures["brain"] = temp_brain
        self.metadata["brain"] = {"classes": ["Normal", "Brain Tumor"], "task": "segmentation_and_screening"}

        # 3. Skin Model
        skin_path = models_dir / "skin" / "best_resnet18.pth"
        skin_model = SkinResNet18(num_classes=8, pretrained=False).to(self.device)
        temp_skin = 1.22
        if skin_path.exists():
            ckpt = torch.load(skin_path, map_location=self.device, weights_only=False)
            skin_model.load_state_dict(ckpt["model_state_dict"])
            temp_skin = ckpt.get("temperature", 1.22)
            proto = ckpt.get("prototypes", {}).get("centroid") if isinstance(ckpt.get("prototypes"), dict) else ckpt.get("feature_prototype_mean")
            if proto is not None:
                self.prototypes["skin"] = torch.tensor(proto, dtype=torch.float32)
        skin_model.eval()
        self.models["skin"] = skin_model
        self.temperatures["skin"] = temp_skin
        self.metadata["skin"] = {"classes": SKIN_CLASSES, "task": "multi_class"}

        # 4. Retina Model
        retina_path = models_dir / "retina" / "best_resnet18.pth"
        retina_model = RetinaResNet18(num_classes=5, pretrained=False).to(self.device)
        temp_retina = 1.18
        if retina_path.exists():
            ckpt = torch.load(retina_path, map_location=self.device, weights_only=False)
            retina_model.load_state_dict(ckpt["model_state_dict"])
            temp_retina = ckpt.get("temperature", 1.18)
            proto = ckpt.get("prototypes", {}).get("centroid") if isinstance(ckpt.get("prototypes"), dict) else ckpt.get("feature_prototype_mean")
            if proto is not None:
                self.prototypes["retina"] = torch.tensor(proto, dtype=torch.float32)
        retina_model.eval()
        self.models["retina"] = retina_model
        self.temperatures["retina"] = temp_retina
        self.metadata["retina"] = {"classes": RETINA_CLASSES, "task": "multi_class"}

        self.is_loaded = True
        print("[TRUSTMED Registry] All 4 specialty models loaded successfully!")

def get_registry() -> ModelRegistry:
    return ModelRegistry.get_instance()
