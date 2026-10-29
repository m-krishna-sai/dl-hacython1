"""
TRUSTMED Human Specialist Recommendation Engine
Maps detected specialty and reliability signals to appropriate medical specialist disciplines
and computes clinical review priority.
Includes required safety notices and decision-support disclaimers.
"""

from typing import Dict, Any

SPECIALTY_MAPPING = {
    "lung": {
        "specialist_title": "Pulmonologist / Thoracic Radiologist",
        "department": "Pulmonology & Thoracic Imaging",
        "clinical_focus": "Evaluation of thoracic opacity, consolidation, effusion, and parenchymal lung findings."
    },
    "brain": {
        "specialist_title": "Neurologist / Neurosurgeon / Neuroradiologist",
        "department": "Neuro-Oncology & Neuroradiology",
        "clinical_focus": "Evaluation of cranial MRI signal abnormalities, space-occupying lesions, and intracranial mass effect."
    },
    "skin": {
        "specialist_title": "Dermatologist / Dermato-Oncologist",
        "department": "Dermatology",
        "clinical_focus": "Dermoscopic examination, lesion margin asymmetry, pigment distribution, and biopsy consideration."
    },
    "retina": {
        "specialist_title": "Ophthalmologist / Retinal Specialist",
        "department": "Ophthalmology & Vitreoretinal Surgery",
        "clinical_focus": "Funduscopic grading of microvascular diabetic retinopathy, macular edema, and neovascularization."
    }
}

class SpecialistRecommender:
    def recommend(
        self,
        specialty: str,
        reliability_level: str,
        finding_severity: str = "normal"
    ) -> Dict[str, Any]:
        """
        Determines specialist referral department and review urgency priority.
        """
        spec_info = SPECIALTY_MAPPING.get(specialty, {
            "specialist_title": "General Medical Practitioner / Triage Specialist",
            "department": "Diagnostic Triage",
            "clinical_focus": "Clinical assessment and referral determination."
        })

        # Urgency Priority Calculation:
        # If AI Reliability is LOW -> Urgent/High Priority Human Review Required
        # If finding is high severity (e.g. Mass, Tumor, Severe DR, Melanoma) -> High Priority
        is_urgent = (reliability_level == "LOW") or (finding_severity in ["severe", "critical", "malignant"])

        if is_urgent:
            review_priority = "HIGH PRIORITY HUMAN REVIEW"
            urgency_class = "urgent"
            action_prompt = (
                "⚠️ Due to low AI reliability indicators or significant clinical severity, "
                "this case must be reviewed promptly by a licensed human specialist. "
                "Do NOT rely on automated outputs."
            )
        elif reliability_level == "MEDIUM":
            review_priority = "ROUTINE SPECIALIST CONSULTATION"
            urgency_class = "moderate"
            action_prompt = "AI-assisted decision-support available. Secondary verification by attending specialist advised."
        else:
            review_priority = "STANDARD CLINICAL REVIEW"
            urgency_class = "standard"
            action_prompt = "AI reliability indicators are stable. Corroborate with clinical history and specialist judgment."

        return {
            "recommended_specialist": spec_info["specialist_title"],
            "department": spec_info["department"],
            "clinical_focus": spec_info["clinical_focus"],
            "review_priority": review_priority,
            "urgency_class": urgency_class,
            "action_prompt": action_prompt,
            "safety_notice": (
                "TrustMed is an investigational AI-assisted decision-support prototype. "
                "It does not provide definitive medical diagnoses, prescriptions, or treatment plans. "
                "All clinical actions remain the sole responsibility of qualified medical professionals."
            )
        }

specialist_recommender = SpecialistRecommender()
