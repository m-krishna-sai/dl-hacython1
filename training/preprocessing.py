"""
TRUSTMED Preprocessing and Augmentation Pipeline
Provides standardized ImageNet-normalized transforms for classification (224x224)
and synchronized spatial transforms for BraTS MRI images and masks.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random
from typing import Tuple, Optional

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_classification_transforms(is_train: bool = True, input_size: Tuple[int, int] = (224, 224), allow_hflip: bool = True) -> T.Compose:
    """
    Returns transforms for medical classification models.
    Mild, medically appropriate augmentations during training.
    """
    if is_train:
        transform_list = [
            T.Resize(input_size),
            T.RandomRotation(degrees=(-10, 10)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if allow_hflip:
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return T.Compose(transform_list)
    else:
        return T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class SynchronizedSegmentationTransform:
    """
    Applies identical spatial transformations to both Brain MRI image and tumor mask.
    Preserves mask binary alignment and avoids interpolation artifacts on segmentation labels.
    """
    def __init__(self, is_train: bool = True, input_size: Tuple[int, int] = (224, 224)):
        self.is_train = is_train
        self.input_size = input_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize image with bilinear, mask with nearest neighbor to preserve discrete values
        image = TF.resize(image, self.input_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.input_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random slight rotation (-10 to 10 deg)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Convert image to tensor and normalize
        img_t = TF.to_tensor(image)
        if img_t.shape[0] == 1:
            img_t = img_t.repeat(3, 1, 1)
        elif img_t.shape[0] > 3:
            img_t = img_t[:3, :, :]
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Convert mask to binary float tensor [1, H, W]
        mask_t = TF.to_tensor(mask)
        if mask_t.shape[0] > 1:
            mask_t = mask_t[0:1, :, :]
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t

def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses ImageNet normalization for Grad-CAM overlay and visualization.
    Input shape: [C, H, W] or [B, C, H, W] in range [0, 1] approximately.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
