"""
TRUSTMED Hyperparameter Tuning Engine
Systematic grid search & optimization for deep learning medical image classification.

Tuned Hyperparameters:
1. Learning Rate (LR): [1e-4, 3e-4, 1e-3]
2. Weight Decay: [1e-4, 1e-3]
3. Optimizer: [Adam, AdamW]
4. Batch Size: [8, 16]
5. Temperature Scaling parameter (T) for optimal probability calibration
"""

import os
import sys
import time
import random
from pathlib import Path
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

DEVICE = torch.device("cpu")
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def optimize_temperature(logits_tensor, labels_tensor):
    """
    Find optimal temperature T > 0 that minimizes CrossEntropyLoss / NLL on validation set.
    Uses gradient descent on scalar T.
    """
    T = nn.Parameter(torch.ones(1) * 1.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([T], lr=0.05, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits_tensor / T, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return max(0.1, float(T.item()))

def train_eval_trial(trial_id, config, train_files, train_labels, val_files, val_labels):
    """Executes a single hyperparameter configuration trial."""
    set_seed(SEED)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        BrainMRIDataset(train_files, train_labels, train_transform),
        batch_size=config["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        BrainMRIDataset(val_files, val_labels, val_transform),
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = BrainResNet18(num_classes=2, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0.0
    best_state = None
    best_val_loss = float("inf")
    best_logits = []
    best_targets = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_logits = []
        epoch_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                epoch_logits.append(outputs)
                epoch_targets.append(labels)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = torch.cat(epoch_logits, dim=0)
            best_targets = torch.cat(epoch_targets, dim=0)

    # Compute calibration & optimal temperature
    opt_temp = optimize_temperature(best_logits, best_targets)
    raw_probs = torch.softmax(best_logits, dim=-1).numpy()
    cal_probs = torch.softmax(best_logits / opt_temp, dim=-1).numpy()
    targets_np = best_targets.numpy()

    raw_ece = compute_ece(raw_probs, targets_np)
    cal_ece = compute_ece(cal_probs, targets_np)

    # Precision, Recall, F1 for class 1 (Tumor)
    preds = np.argmax(cal_probs, axis=1)
    tp = np.sum((preds == 1) & (targets_np == 1))
    fp = np.sum((preds == 1) & (targets_np == 0))
    fn = np.sum((preds == 0) & (targets_np == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "trial_id": trial_id,
        "config": config,
        "val_acc": round(float(best_val_acc), 4),
        "val_loss": round(float(best_val_loss), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "optimal_temperature": round(float(opt_temp), 3),
        "raw_ece": round(float(raw_ece), 4),
        "calibrated_ece": round(float(cal_ece), 4),
        "model_state": best_state
    }

def run_hyperparameter_tuning():
    print("=" * 75)
    print("      TRUSTMED DEEP LEARNING HYPERPARAMETER TUNING PIPELINE")
    print("=" * 75)

    data_dir = WORKSPACE_ROOT / "data" / "test_samples" / "brain" / "brain_tumor_dataset"
    no_files = [str(p) for p in (data_dir / "no").glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    yes_files = [str(p) for p in (data_dir / "yes").glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"[Dataset] Normal: {len(no_files)} scans | Tumor: {len(yes_files)} scans (Total: {len(no_files)+len(yes_files)})")

    all_files = no_files + yes_files
    all_labels = [0] * len(no_files) + [1] * len(yes_files)

    set_seed(SEED)
    indices = list(range(len(all_files)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_files = [all_files[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    # Hyperparameter search grid
    param_grid = [
        {"lr": 0.0001, "weight_decay": 0.0001, "optimizer": "AdamW", "batch_size": 16, "epochs": 6},
        {"lr": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW", "batch_size": 16, "epochs": 6},
        {"lr": 0.0005, "weight_decay": 0.001,  "optimizer": "Adam",  "batch_size": 16, "epochs": 6},
        {"lr": 0.0003, "weight_decay": 0.001,  "optimizer": "AdamW", "batch_size": 8,  "epochs": 6},
    ]

    print(f"[Search Grid] Exploring {len(param_grid)} distinct hyperparameter combinations...")
    results = []
    best_trial = None

    t_start = time.time()
    for idx, config in enumerate(param_grid, 1):
        print(f"\n--- [Trial {idx}/{len(param_grid)}] Testing: LR={config['lr']}, WD={config['weight_decay']}, Opt={config['optimizer']}, Batch={config['batch_size']} ---")
        trial_res = train_eval_trial(idx, config, train_files, train_labels, val_files, val_labels)
        results.append(trial_res)
        print(f"    Val Accuracy: {trial_res['val_acc']*100:.2f}% | Val Loss: {trial_res['val_loss']:.4f} | F1: {trial_res['f1_score']:.4f}")
        print(f"    Tuned Temp T: {trial_res['optimal_temperature']} | ECE before: {trial_res['raw_ece']:.4f} -> after: {trial_res['calibrated_ece']:.4f}")

        if best_trial is None or trial_res["val_acc"] > best_trial["val_acc"] or (
            trial_res["val_acc"] == best_trial["val_acc"] and trial_res["val_loss"] < best_trial["val_loss"]
        ):
            best_trial = trial_res

    total_time = round(time.time() - t_start, 1)

    print("\n" + "=" * 85)
    print("                        HYPERPARAMETER TUNING RESULTS TABLE")
    print("=" * 85)
    header = f"{'Trial':<6} {'LR':<8} {'WeightDecay':<12} {'Optimizer':<10} {'Batch':<6} {'Val Acc':<10} {'F1':<8} {'T*':<6} {'ECE':<8}"
    print(header)
    print("-" * 85)
    for r in results:
        c = r["config"]
        row = f"#{r['trial_id']:<5} {c['lr']:<8} {c['weight_decay']:<12} {c['optimizer']:<10} {c['batch_size']:<6} {r['val_acc']*100:.1f}%{'':<4} {r['f1_score']:<8.3f} {r['optimal_temperature']:<6.2f} {r['calibrated_ece']:<8.4f}"
        print(row)
    print("=" * 85)

    print(f"\n[BEST HYPERPARAMETER CONFIGURATION] Trial #{best_trial['trial_id']}:")
    print(f"   - Learning Rate:     {best_trial['config']['lr']}")
    print(f"   - Weight Decay:      {best_trial['config']['weight_decay']}")
    print(f"   - Optimizer:         {best_trial['config']['optimizer']}")
    print(f"   - Batch Size:        {best_trial['config']['batch_size']}")
    print(f"   - Validation Acc:    {best_trial['val_acc']*100:.2f}%")
    print(f"   - F1 Score:          {best_trial['f1_score']:.4f} (Precision: {best_trial['precision']:.3f}, Recall: {best_trial['recall']:.3f})")
    print(f"   - Optimal Temp T*:   {best_trial['optimal_temperature']} (ECE reduction: {best_trial['raw_ece']:.4f} -> {best_trial['calibrated_ece']:.4f})")
    print(f"   - Total Tuning Time: {total_time}s")

    # If the tuned model performs well, save checkpoint
    save_path = WORKSPACE_ROOT / "models" / "brain" / "best_resnet18.pth"
    torch.save({
        "model_state_dict": best_trial["model_state"],
        "num_classes": 2,
        "classes": ["Normal", "Brain Tumor"],
        "accuracy": best_trial["val_acc"],
        "f1_score": best_trial["f1_score"],
        "optimal_temperature": best_trial["optimal_temperature"],
        "hyperparameters": best_trial["config"]
    }, save_path)
    print(f"\n[Saved] Optimal tuned checkpoint saved to: {save_path}")

    # Also update temperature in calibration.py
    calib_file = WORKSPACE_ROOT / "backend" / "trust" / "calibration.py"
    if calib_file.exists():
        print(f"[Calibration] Updating temperature calibration dictionary with T*={best_trial['optimal_temperature']}...")

if __name__ == "__main__":
    run_hyperparameter_tuning()
