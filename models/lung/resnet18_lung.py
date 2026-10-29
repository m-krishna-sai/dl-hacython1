"""
TRUSTMED Lung Model — ResNet-18 Multi-Label Thoracic Disease Classifier
Supports NIH ChestX-ray14 14 findings, MC Dropout, Penultimate Feature Extraction,
and Optional CBAM Attention.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from models.attention import CBAMBlock, SEBlock

LUNG_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

class LungResNet18(nn.Module):
    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.3, use_attention: bool = False, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAMBlock(in_planes=512)
        else:
            self.attention = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 512-dim feature embedding for OOD detection."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.forward_features(x)
        dropped = self.dropout(feat)
        logits = self.fc(dropped)
        if return_features:
            return logits, feat
        return logits

    def enable_mc_dropout(self):
        """Forces dropout to remain active during evaluation mode for MC Dropout."""
        self.dropout.train()
