"""
TRUSTMED Brain U-Net — Standard 2D Segmentation Model
Segments brain tumors / abnormalities on MRI slices (BraTS 2021).
Calculates Dice coefficient and IoU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu layers."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class BrainUNet(nn.Module):
    """
    Standard U-Net with contracting path (encoder), bottleneck,
    expanding path (decoder) with skip connections, and segmentation head.
    Optimized for laptop CPU execution.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 4, f * 8)

        # Decoder (Expanding Path) with Skip Connections
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        # Output head
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Logits
        return self.out_conv(d1)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Outputs binary segmentation mask [B, 1, H, W] from input tensor."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

def compute_dice_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6):
    """
    Computes Dice coefficient and IoU between prediction and target.
    Expects binary tensors of shape [B, 1, H, W].
    """
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return float(dice.item()), float(iou.item())
