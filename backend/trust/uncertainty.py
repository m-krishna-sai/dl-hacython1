"""
TRUSTMED Monte Carlo Dropout Uncertainty Engine
Performs repeated stochastic forward passes with dropout active at test time
to quantify epistemic (model) uncertainty.
Calculates variance across stochastic runs, Shannon entropy, and dispersion.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

class MCDropoutUncertainty:
    def __init__(self, num_passes: int = 10):
        self.num_passes = num_passes

    def evaluate(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        task: str = "multi_class"
    ) -> Dict[str, Any]:
        """
        Executes stochastic forward passes with active dropout.
        input_tensor: [1, 3, 224, 224]
        Returns variance, entropy, individual run predictions, and uncertainty category.
        """
        # Enable dropout during inference
        if hasattr(model, "enable_mc_dropout"):
            model.enable_mc_dropout()
        else:
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_passes):
                logits = model(input_tensor)
                if task == "multi_label":
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                else:
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                predictions.append(probs)

        # Restore eval mode
        model.eval()

        predictions = np.array(predictions) # Shape: [num_passes, num_classes]

        # Variance across passes per class
        variance_per_class = np.var(predictions, axis=0)
        mean_variance = float(np.mean(variance_per_class))
        max_variance = float(np.max(variance_per_class))

        # Mean predicted probabilities
        mean_probs = np.mean(predictions, axis=0)

        # Shannon entropy of mean prediction
        eps = 1e-8
        if task == "multi_label":
            entropy = float(-np.mean(mean_probs * np.log(mean_probs + eps) + (1 - mean_probs) * np.log(1 - mean_probs + eps)))
        else:
            entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))

        # Categorize uncertainty
        # Typically, variance < 0.005 is low, 0.005-0.02 is medium, > 0.02 is high
        if mean_variance < 0.006 and entropy < 1.0:
            uncertainty_level = "LOW"
            description = "Stochastic predictions are tightly clustered. Low epistemic uncertainty."
        elif mean_variance < 0.020:
            uncertainty_level = "MEDIUM"
            description = "Moderate dispersion across stochastic runs. Moderate model hesitation."
        else:
            uncertainty_level = "HIGH"
            description = "High predictive variance across stochastic passes. Significant model hesitation."

        # Sample run outputs for transparency in UI
        run_summaries = []
        for i in range(min(5, self.num_passes)):
            top_prob = float(np.max(predictions[i]))
            run_summaries.append({
                "pass": i + 1,
                "top_probability": round(top_prob, 4),
                "top_class_idx": int(np.argmax(predictions[i]))
            })

        return {
            "uncertainty_level": uncertainty_level,
            "variance": round(mean_variance, 6),
            "max_class_variance": round(max_variance, 6),
            "entropy": round(entropy, 4),
            "num_stochastic_passes": self.num_passes,
            "description": description,
            "stochastic_runs_sample": run_summaries,
            "mean_probabilities": [round(float(p), 4) for p in mean_probs]
        }

mc_uncertainty = MCDropoutUncertainty(num_passes=10)
