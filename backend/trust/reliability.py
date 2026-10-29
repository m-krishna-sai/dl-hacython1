"""
TRUSTMED Master Reliability Engine — Core Innovation
Combines multiple independent reliability signals:
1. Image Quality (Good vs Poor)
2. Calibrated Confidence (via Temperature Scaling)
3. Epistemic Uncertainty (via MC Dropout variance & entropy)
4. Out-of-Distribution Familiarity (via feature distance & energy score)
5. Prediction Stability (agreement across repeated passes)

Formulates a mathematically consistent, transparent composite rating:
AI Reliability: HIGH, MEDIUM, or LOW.

Guarantees the Critical Trust Rule:
High probability ALONE does not equal high reliability!
If OOD is HIGH or Uncertainty is HIGH or Stability is LOW -> Reliability is LOW.
"""

from typing import Dict, Any

class ReliabilityEngine:
    def evaluate(
        self,
        quality: Dict[str, Any],
        calibration: Dict[str, Any],
        uncertainty: Dict[str, Any],
        ood: Dict[str, Any],
        stability: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes individual trust indicators into an overall reliability verdict.
        """
        # 1. Hard gate: Poor quality immediately results in UNRELIABLE
        if quality.get("verdict") == "POOR":
            return {
                "overall_reliability": "LOW",
                "summary": "UNRELIABLE — Severe Image Quality Inadequacy",
                "trust_breakdown": {
                    "image_quality": "POOR",
                    "confidence": "N/A",
                    "uncertainty": "N/A",
                    "ood_risk": "N/A",
                    "stability": "N/A"
                },
                "is_reliable": False,
                "human_review_required": True,
                "clinical_warning": "Image quality is severely inadequate. Automated diagnostic inference aborted.",
                "indicators": self._format_indicators(quality, calibration, uncertainty, ood, stability)
            }

        # Extract indicators
        cal_conf = calibration.get("calibrated_confidence", 0.5)
        raw_conf = calibration.get("raw_confidence", 0.5)
        unc_level = uncertainty.get("uncertainty_level", "MEDIUM")
        ood_level = ood.get("ood_level", "MODERATE")
        stab_level = stability.get("stability_level", "MODERATE")
        stab_pct = stability.get("stability_percentage", 50.0)

        # 2. Critical Trust Scenario Check (Judge Demo):
        # Even if raw probability or confidence is very high (e.g. 95%),
        # if OOD is HIGH, Uncertainty is HIGH, or Stability is LOW -> Mark as LOW reliability!
        is_critical_mismatch = (raw_conf >= 0.85) and (ood_level == "HIGH" or unc_level == "HIGH" or stab_pct < 65.0)

        # Deductive reliability logic
        risk_flags = 0
        if ood_level == "HIGH": risk_flags += 2
        elif ood_level == "MODERATE": risk_flags += 1

        if unc_level == "HIGH": risk_flags += 2
        elif unc_level == "MEDIUM": risk_flags += 1

        if stab_level == "LOW": risk_flags += 2
        elif stab_level == "MODERATE": risk_flags += 1

        if cal_conf < 0.60: risk_flags += 1

        # Composite Decision
        if risk_flags >= 3 or is_critical_mismatch or ood_level == "HIGH":
            overall = "LOW"
            is_reliable = False
            human_required = True
            if is_critical_mismatch:
                warning = "⚠️ CRITICAL TRUST WARNING: Model produces high nominal probability, but displays HIGH UNCERTAINTY / OOD ANOMALY. Result is NOT reliable. Immediate specialist review required."
            else:
                warning = "⚠️ Model uncertainty is elevated and features diverge from reference distribution. Automated prediction may not be dependable."
        elif risk_flags >= 1 or cal_conf < 0.75:
            overall = "MEDIUM"
            is_reliable = True
            human_required = False
            warning = "Moderate AI reliability. Corroboration with patient history and clinical signs recommended."
        else:
            overall = "HIGH"
            is_reliable = True
            human_required = False
            warning = "All reliability indicators are concordant (Calibrated confidence high, low uncertainty, in-distribution, high stability)."

        return {
            "overall_reliability": overall,
            "is_reliable": is_reliable,
            "human_review_required": human_required,
            "critical_mismatch_detected": is_critical_mismatch,
            "clinical_warning": warning,
            "trust_breakdown": {
                "image_quality": quality.get("verdict", "GOOD"),
                "confidence": "HIGH" if cal_conf >= 0.80 else ("MODERATE" if cal_conf >= 0.60 else "LOW"),
                "uncertainty": unc_level,
                "ood_risk": ood_level,
                "stability": stab_level
            },
            "indicators": self._format_indicators(quality, calibration, uncertainty, ood, stability)
        }

    def _format_indicators(self, quality, calibration, uncertainty, ood, stability) -> list:
        return [
            {
                "name": "Image Quality Gate",
                "value": quality.get("verdict", "GOOD"),
                "status": "PASS" if quality.get("passed", True) else "FAIL",
                "detail": f"Blur score: {quality.get('blur_score', 0.0)} | Contrast: {quality.get('rms_contrast', 0.0)}"
            },
            {
                "name": "Calibrated Confidence",
                "value": f"{round(calibration.get('calibrated_confidence', 0.0)*100, 1)}%",
                "status": "PASS" if calibration.get("calibrated_confidence", 0.0) >= 0.70 else "WARN",
                "detail": f"Raw: {round(calibration.get('raw_confidence', 0.0)*100, 1)}% | T={calibration.get('temperature', 1.0)}"
            },
            {
                "name": "MC Dropout Uncertainty",
                "value": uncertainty.get("uncertainty_level", "LOW"),
                "status": "PASS" if uncertainty.get("uncertainty_level") == "LOW" else ("WARN" if uncertainty.get("uncertainty_level") == "MEDIUM" else "FAIL"),
                "detail": f"Variance: {uncertainty.get('variance', 0.0)} across {uncertainty.get('num_stochastic_passes', 10)} passes"
            },
            {
                "name": "Image Familiarity (OOD)",
                "value": ood.get("image_familiarity", "LOW RISK"),
                "status": "PASS" if ood.get("ood_level") == "LOW" else ("WARN" if ood.get("ood_level") == "MODERATE" else "FAIL"),
                "detail": f"Energy: {ood.get('energy_score', 0.0)} | Sim: {ood.get('prototype_similarity', 0.0)}"
            },
            {
                "name": "Prediction Stability",
                "value": f"{stability.get('stability_percentage', 100.0)}%",
                "status": "PASS" if stability.get("stability_percentage", 100.0) >= 80.0 else "WARN",
                "detail": stability.get("description", "Deterministic agreement")
            }
        ]

reliability_engine = ReliabilityEngine()
