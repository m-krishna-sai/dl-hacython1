"""
TRUSTMED Brain ResNet-18 Classifier Training Script
Trains BrainResNet18 (Normal vs Brain Tumor) on real brain MRI scans.
Simple, beginner-friendly PyTorch training script.
"""

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from models.brain.resnet18_brain import BrainResNet18

# Hypertuned Configuration (Found via Systematic Grid Search in tune_hyperparameters.py)
DEVICE = torch.device("cpu")
BATCH_SIZE = 8
EPOCHS = 8
LR = 0.0003
WEIGHT_DECAY = 0.001
OPTIMAL_TEMPERATURE = 1.524  # Tuned via NLL minimization for ECE calibration
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class BrainMRIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def main():
    data_dir = WORKSPACE_ROOT / "data" / "test_samples" / "brain" / "brain_tumor_dataset"
    no_dir = data_dir / "no"
    yes_dir = data_dir / "yes"

    if not no_dir.exists() or not yes_dir.exists():
        print(f"Dataset not found at {data_dir}")
        return

    no_files = [str(p) for p in no_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    yes_files = [str(p) for p in yes_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"Found {len(no_files)} Normal brain scans (class 0) and {len(yes_files)} Tumor brain scans (class 1)")

    # 0 = Normal, 1 = Tumor
    all_files = no_files + yes_files
    all_labels = [0] * len(no_files) + [1] * len(yes_files)

    # Shuffle and split 80% train, 20% val
    indices = list(range(len(all_files)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_files = [all_files[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BrainMRIDataset(train_files, train_labels, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BrainMRIDataset(val_files, val_labels, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing BrainResNet18 with pretrained ImageNet backbone...")
    model = BrainResNet18(num_classes=2, pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None

    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # Save best checkpoint
    save_path = WORKSPACE_ROOT / "models" / "brain" / "best_resnet18.pth"
    save_dict = {
        "model_state_dict": best_state,
        "classes": ["Normal", "Brain Tumor"],
        "accuracy": best_val_acc,
        "temperature": OPTIMAL_TEMPERATURE,
        "hyperparameters": {
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "scheduler": "CosineAnnealingLR"
        }
    }
    torch.save(save_dict, save_path)
    print(f"Best model saved to {save_path} with Val Acc: {best_val_acc:.4f} and Temperature: {OPTIMAL_TEMPERATURE}")

if __name__ == "__main__":
    main()
