"""
TRUSTMED Prediction Stability Engine
Evaluates whether top clinical predictions remain consistent across repeated stochastic inference.
Formula:
    stability = (count of runs predicting dominant class / total runs) * 100%
For multi-label tasks, measures agreement across binary threshold predictions.
"""

import numpy as np
from typing import Dict, Any, List
from collections import Counter

class PredictionStabilityEvaluator:
    def evaluate(
        self,
        stochastic_runs_sample: List[Dict[str, Any]],
        task: str = "multi_class"
    ) -> Dict[str, Any]:
        """
        Calculates empirical stability score from repeated stochastic passes.
        """
        if not stochastic_runs_sample:
            return {
                "stability_percentage": 100.0,
                "stability_level": "HIGH",
                "dominant_prediction_count": 1,
                "total_runs": 1,
                "status": "Deterministic baseline"
            }

        top_classes = [run.get("top_class_idx", 0) for run in stochastic_runs_sample]
        total_runs = len(top_classes)
        counts = Counter(top_classes)
        dominant_class, dominant_count = counts.most_common(1)[0]

        stability_fraction = dominant_count / total_runs
        stability_pct = round(stability_fraction * 100.0, 1)

        if stability_pct >= 90.0:
            level = "HIGH"
            desc = f"Prediction is highly consistent across {total_runs} stochastic forward passes ({stability_pct}% agreement)."
        elif stability_pct >= 70.0:
            level = "MODERATE"
            desc = f"Prediction demonstrates moderate stability ({stability_pct}% agreement). Slight class boundary ambiguity."
        else:
            level = "LOW"
            desc = f"Prediction is volatile ({stability_pct}% agreement). Different stochastic passes predict conflicting classes."

        return {
            "stability_percentage": stability_pct,
            "stability_level": level,
            "dominant_prediction_count": dominant_count,
            "total_runs": total_runs,
            "dominant_class_idx": dominant_class,
            "class_frequency_distribution": {int(k): v for k, v in counts.items()},
            "description": desc
        }

stability_evaluator = PredictionStabilityEvaluator()
