"""
TRUSTMED Clinical & Reliability Metrics Evaluation Suite
Computes:
- Classification: Macro-AUROC, Sensitivity, Specificity, Accuracy, Precision, Recall, Confusion Matrix
- Segmentation: Dice Coefficient, IoU (Intersection over Union)
- Reliability & Calibration: Expected Calibration Error (ECE), Brier Score, MC Dropout Variance, Prediction Stability
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    brier_score_loss
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task: str = "binary",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics for binary, multi-class, or multi-label tasks.
    y_true: [N] or [N, C]
    y_prob: [N] or [N, C] in range [0, 1]
    """
    metrics = {}

    if task == "multi_label":
        # Multi-label classification (e.g. Lung ChestX-ray14)
        n_classes = y_true.shape[1]
        aurocs = []
        sensitivities = []
        specificities = []
        precisions = []
        recalls = []

        for c in range(n_classes):
            yt_c = y_true[:, c]
            yp_c = y_prob[:, c]
            y_pred_c = (yp_c >= threshold).astype(int)

            if len(np.unique(yt_c)) > 1:
                try:
                    aurocs.append(roc_auc_score(yt_c, yp_c))
                except Exception:
                    pass

            # Confusion matrix for class c
            cm = confusion_matrix(yt_c, y_pred_c, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            sensitivities.append(sens)
            specificities.append(spec)
            precisions.append(prec)
            recalls.append(sens)

        metrics["macro_auroc"] = round(float(np.mean(aurocs)), 4) if aurocs else None
        metrics["sensitivity"] = round(float(np.mean(sensitivities)), 4)
        metrics["specificity"] = round(float(np.mean(specificities)), 4)
        metrics["precision"] = round(float(np.mean(precisions)), 4)
        metrics["recall"] = round(float(np.mean(recalls)), 4)
        metrics["accuracy"] = round(float(np.mean((y_prob >= threshold) == y_true)), 4)

    elif task in ("multi_class", "binary"):
        # Multi-class or binary classification (e.g. Skin, Retina, Brain classifier)
        if task == "binary":
            y_pred = (y_prob >= threshold).astype(int)
            if len(np.unique(y_true)) > 1:
                metrics["macro_auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            else:
                metrics["macro_auroc"] = None

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = round(float(tp / (tp + fn) if (tp + fn) > 0 else 0.0), 4)
            metrics["specificity"] = round(float(tn / (tn + fp) if (tn + fp) > 0 else 0.0), 4)
            metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            metrics["recall"] = metrics["sensitivity"]
            metrics["confusion_matrix"] = cm.tolist()
        else:
            y_pred = np.argmax(y_prob, axis=1)
            metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            metrics["precision"] = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            metrics["sensitivity"] = metrics["recall"]

            try:
                metrics["macro_auroc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4)
            except Exception:
                metrics["macro_auroc"] = None

            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # Per-class specificity macro average
            specs = []
            for i in range(len(cm)):
                tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                fp = np.sum(np.delete(cm[:, i], i))
                specs.append(tn / (tn + fp) if (tn + fp) > 0 else 1.0)
            metrics["specificity"] = round(float(np.mean(specs)), 4)

    return metrics

def compute_segmentation_metrics(
    y_true_mask: np.ndarray,
    y_pred_mask: np.ndarray,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes Dice coefficient and IoU for binary segmentation.
    y_true_mask, y_pred_mask: boolean or {0, 1} arrays of any shape.
    """
    true_f = y_true_mask.flatten() > 0.5
    pred_f = y_pred_mask.flatten() > 0.5

    intersection = np.logical_and(true_f, pred_f).sum()
    dice = (2.0 * intersection + smooth) / (true_f.sum() + pred_f.sum() + smooth)
    union = np.logical_or(true_f, pred_f).sum()
    iou = (intersection + smooth) / (union + smooth)

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4)
    }

def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculates Expected Calibration Error (ECE) and bin reliability diagram data.
    probs: predicted probability of top class [N]
    labels: binary indicator if top class is correct (1) or incorrect (0) [N]
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        in_bin = (probs > b_low) & (probs <= b_high)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(labels[in_bin])
            conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_data.append({
                "bin": f"{b_low:.1f}-{b_high:.1f}",
                "confidence": round(float(conf_in_bin), 4),
                "accuracy": round(float(acc_in_bin), 4),
                "count": int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                "bin": f"{b_low:.1f}-{b_high:.1f}",
                "confidence": round(float((b_low + b_high) / 2), 4),
                "accuracy": round(float((b_low + b_high) / 2), 4),
                "count": 0
            })

    return {
        "ece": round(float(ece), 4),
        "bin_reliability_diagram": bin_data
    }
