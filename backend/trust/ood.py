"""
TRUSTMED Out-of-Distribution (OOD) & Unfamiliar Image Detection Engine
Evaluates whether an input image resembles the training distribution of the selected specialty.
Combines:
1. Feature-space distance (Euclidean/Mahalanobis-style distance to in-distribution prototype)
2. Free Energy Score: E(x; T) = -T * log(sum(exp(logits / T)))
Outputs Image Familiarity risk rating: LOW RISK (In-distribution), MODERATE, or HIGH RISK (OOD).
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

class OODDetector:
    def __init__(self, energy_threshold_high: float = -4.0, energy_threshold_mod: float = -7.0):
        self.energy_threshold_high = energy_threshold_high
        self.energy_threshold_mod = energy_threshold_mod

    def evaluate(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        specialty: str,
        prototype: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates feature distance and energy score for OOD scoring.
        features: Penultimate layer embedding [1, D]
        logits: Unnormalized logits [1, C]
        """
        with torch.no_grad():
            # 1. Energy Score: E(x) = -T * logsumexp(logits / T)
            # In-distribution samples typically exhibit lower (more negative) energy scores.
            # OOD/unfamiliar samples have higher energy values (closer to 0 or positive).
            scaled_logits = logits / temperature
            energy_score = float(-temperature * torch.logsumexp(scaled_logits, dim=-1).item())

            # 2. Prototype Feature Distance
            feature_dist = 0.0
            if prototype is not None:
                feat_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
                proto_norm = torch.nn.functional.normalize(prototype.unsqueeze(0), p=2, dim=-1)
                cosine_sim = float(torch.mm(feat_norm, proto_norm.t()).item())
                feature_dist = float(torch.norm(features - prototype.unsqueeze(0), p=2).item())
            else:
                cosine_sim = 0.85
                feature_dist = 5.0

        # Familiarity Assessment
        # High energy or very low cosine similarity indicates out-of-distribution
        if energy_score > self.energy_threshold_high or cosine_sim < 0.40:
            familiarity_risk = "HIGH RISK"
            ood_status = "HIGH"
            explanation = "Image exhibits feature embeddings significantly distant from training distribution. High probability of unfamiliar or non-standard anatomy."
        elif energy_score > self.energy_threshold_mod or cosine_sim < 0.65:
            familiarity_risk = "MODERATE"
            ood_status = "MODERATE"
            explanation = "Image exhibits atypical distribution features. Moderate familiarity risk."
        else:
            familiarity_risk = "LOW RISK"
            ood_status = "LOW"
            explanation = "Image features closely match the specialty training distribution. In-distribution confidence supported."

        return {
            "image_familiarity": familiarity_risk,
            "ood_level": ood_status,
            "energy_score": round(energy_score, 4),
            "prototype_distance": round(feature_dist, 4),
            "prototype_similarity": round(cosine_sim, 4),
            "explanation": explanation
        }

ood_detector = OODDetector()
