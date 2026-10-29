"""
TRUSTMED Clinical PDF Report Generator
Generates downloadable, publication-ready PDF medical decision-support reports
using ReportLab.
Includes:
- Patient/Study Metadata & Timestamp
- Detected Specialty & Medical Modality
- Image Quality Gate Indicators
- Diagnostic Predictions & Calibrated Probabilities
- Self-Aware Trust Engine Signals (Calibrated Confidence, MC Uncertainty, OOD Familiarity, Stability)
- Visual Diagnostic Overlays (Grad-CAM or Brain Tumor Segmentation)
- Recommended Specialist & Human Review Urgency
- Medical Safety & Regulatory Disclaimers
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image

def generate_pdf_report(analysis_result: Dict[str, Any]) -> bytes:
    """
    Constructs a professional PDF document in-memory and returns raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0284c7")
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )
    section_heading = ParagraphStyle(
        "SecHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=7,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )
    callout_title = ParagraphStyle(
        "CallTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    callout_body = ParagraphStyle(
        "CallBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # Header Banner
    story.append(Paragraph("TRUSTMED", title_style))
    story.append(Paragraph("Medical Image Assessment", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=6))

    spec_name = analysis_result.get("specialty_name", "General / Undetermined")
    modality = analysis_result.get("modality", "Unknown")
    filename = analysis_result.get("filename", "medical_scan.png")
    q_verdict = analysis_result.get("quality", {}).get("verdict", "GOOD")
    overall_rel = analysis_result.get("reliability", {}).get("overall_reliability", "UNKNOWN")

    rel_color = colors.HexColor("#10b981") if overall_rel == "HIGH" else (
        colors.HexColor("#f59e0b") if overall_rel == "MEDIUM" else colors.HexColor("#ef4444")
    )

    # 1. IMAGE INFORMATION
    story.append(Paragraph("1. IMAGE INFORMATION", section_heading))
    study_data = [
        [
            Paragraph("<b>File:</b> " + filename, body_style),
            Paragraph("<b>Specialty:</b> " + spec_name, body_style)
        ],
        [
            Paragraph("<b>Modality:</b> " + modality, body_style),
            Paragraph(f"<b>Image Quality:</b> <font color='#10b981'><b>{q_verdict}</b></font>", body_style)
        ]
    ]
    t_study = Table(study_data, colWidths=[270, 270])
    t_study.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_study)
    story.append(Spacer(1, 6))

    # 2. AI ASSESSMENT
    story.append(Paragraph("2. AI ASSESSMENT", section_heading))
    pred = analysis_result.get("prediction", {})
    finding_text = pred.get("primary_finding", "No specific diagnostic finding")
    cal = analysis_result.get("calibration", {})

    # Use calibrated confidence as primary confidence
    cal_conf = analysis_result.get("calibrated_confidence")
    if cal_conf is None:
        cal_conf = cal.get("calibrated_confidence", pred.get("top_probability", 0.0))
    conf_pct = f"{round(float(cal_conf)*100, 1)}%"

    rel = analysis_result.get("reliability", {})
    rel_summary = analysis_result.get("reliability_summary")
    if not rel_summary:
        if overall_rel == "HIGH":
            rel_summary = "The AI identified this finding with consistent confidence and familiar image characteristics across Trust Engine validation checks."
        elif overall_rel == "MEDIUM":
            rel_summary = "The AI identified a possible finding with moderate confidence, but additional reliability checks indicate elevated uncertainty that warrants clinical corroboration."
        else:
            rel_summary = "The AI identified this finding with moderate confidence, but additional reliability checks indicate that the result should be interpreted cautiously and may not be dependable enough to use on its own without specialist review."

    pred_data = [
        [Paragraph("<b>AI Finding:</b>", body_style), Paragraph(f"<b>{finding_text}</b>", body_style)],
        [Paragraph("<b>AI Confidence:</b>", body_style), Paragraph(f"<b>{conf_pct}</b> <font color='#64748b' size='7'>(Calibrated probability)</font>", body_style)],
        [Paragraph("<b>AI Reliability:</b>", body_style), Paragraph(f"<font color='{rel_color.hexval()}'><b>{overall_rel}</b></font>", body_style)],
        [Paragraph("<b>What this means:</b>", body_style), Paragraph(rel_summary, body_style)]
    ]

    if rel.get("clinical_warning"):
        pred_data.append([
            Paragraph("<b>Clinical Note:</b>", body_style),
            Paragraph(f"<font color='#dc2626'><b>Caution:</b> {rel['clinical_warning']}</font>", body_style)
        ])

    # Brain tumor segmentation if applicable
    if "segmentation" in pred and pred["segmentation"]:
        seg = pred["segmentation"]
        pred_data.append([
            Paragraph("<b>AI Tumor Segmentation:</b>", body_style),
            Paragraph(f"Identified Region: {'Detected' if seg.get('has_tumor') else 'No distinct tumor detected'} | Coverage: {seg.get('coverage_percentage')}% ({seg.get('tumor_pixels')} px)", body_style)
        ])

    t_pred = Table(pred_data, colWidths=[120, 420])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 6))

    # 3. AI EXPLANATION
    story.append(Paragraph("3. AI EXPLANATION", section_heading))
    img_cells = []
    try:
        orig_b64 = analysis_result.get("image_base64", "")
        if orig_b64 and "," in orig_b64:
            orig_bytes = base64.b64decode(orig_b64.split(",")[1])
            orig_rl = RLImage(io.BytesIO(orig_bytes), width=1.45*inch, height=1.45*inch)
            img_cells.append([orig_rl, Paragraph("<center><b>Original Scan</b></center>", body_style)])
    except Exception:
        pass

    try:
        overlay_b64 = analysis_result.get("gradcam_base64", "")
        overlay_title = "Areas That Influenced the AI"
        if not overlay_b64 and "segmentation" in analysis_result and analysis_result["segmentation"]:
            overlay_b64 = analysis_result["segmentation"].get("overlay_base64", "")
            overlay_title = "AI Tumor Segmentation"

        if overlay_b64 and "," in overlay_b64:
            over_bytes = base64.b64decode(overlay_b64.split(",")[1])
            over_rl = RLImage(io.BytesIO(over_bytes), width=1.45*inch, height=1.45*inch)
            img_cells.append([over_rl, Paragraph(f"<center><b>{overlay_title}</b></center>", body_style)])
    except Exception:
        pass

    if len(img_cells) >= 2:
        t_img = Table([[img_cells[0][0], img_cells[1][0]], [img_cells[0][1], img_cells[1][1]]], colWidths=[270, 270])
        t_img.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_img)
        story.append(Spacer(1, 4))

    # AI assessment rationale (visual patterns only, no biological causal claims)
    rationale_text = analysis_result.get("assessment_rationale")
    if not rationale_text:
        spec = analysis_result.get("specialty", "general")
        if spec == "skin":
            rationale_text = "The model's prediction was influenced by visual characteristics and patterns within the highlighted region, including lesion appearance, pigmentation distribution, overall symmetry, and border characteristics."
        elif spec == "lung":
            rationale_text = "The model's prediction was influenced by regional visual characteristics and focal opacity patterns identified within the highlighted thoracic zones."
        elif spec == "brain":
            rationale_text = "The model's assessment was influenced by localized tissue contrast differences, structural cranial boundaries, and signal intensity variations within the highlighted cranial region."
        elif spec == "retina":
            rationale_text = "The model's prediction was influenced by visual characteristics, retinal vessel architecture, and localized pattern distributions within the highlighted fundus area."
        else:
            rationale_text = "The model's prediction was influenced by visual patterns and regional characteristics within the highlighted area of the image."

    # Educational context (educational only, no individual causality, no prescriptions)
    edu_text = analysis_result.get("educational_context")
    if not edu_text:
        spec = analysis_result.get("specialty", "general")
        if spec == "skin":
            edu_text = "This type of skin finding can have multiple contributing factors. An image alone cannot determine what caused a particular lesion in an individual person. A qualified dermatologist should consider the image together with clinical history, dermoscopic examination, and other relevant context."
        elif spec == "lung":
            edu_text = "Thoracic and pulmonary findings can be associated with a wide variety of clinical and environmental factors. An imaging scan alone cannot determine what caused a specific finding in an individual patient. A qualified pulmonologist or physician should evaluate this scan alongside full medical history, symptoms, and clinical examination."
        elif spec == "brain":
            edu_text = "Cranial imaging findings and space-occupying lesions can be associated with diverse cellular, vascular, or physiological factors. An imaging scan alone cannot establish the underlying cause for an individual patient. A neurologist or neurosurgeon should evaluate these findings in conjunction with clinical history, neurological examination, and dedicated diagnostic workup."
        elif spec == "retina":
            edu_text = "Retinal microvascular findings are commonly associated with systemic vascular and metabolic factors over time. An image alone cannot determine individual etiology or systemic progression. A qualified ophthalmologist or retina specialist should assess the findings alongside clinical history, laboratory results, and comprehensive eye examination."
        else:
            edu_text = "Medical imaging findings can be associated with multiple clinical and physiological factors. An image alone cannot determine the underlying cause for an individual person. A qualified medical specialist should evaluate the findings in the context of complete patient history and clinical examination."

    explanation_box_data = [
        [Paragraph("<b>Why did the AI make this assessment?</b>", callout_title)],
        [Paragraph(rationale_text, callout_body)],
        [Paragraph("<b>About This Finding (What Can Be Associated With This Finding?):</b>", callout_title)],
        [Paragraph(edu_text, callout_body)]
    ]
    t_exp_box = Table(explanation_box_data, colWidths=[540])
    t_exp_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_exp_box)
    story.append(Spacer(1, 6))

    # 4. RECOMMENDED REVIEW
    story.append(Paragraph("4. RECOMMENDED REVIEW", section_heading))
    spec = analysis_result.get("specialist", {})
    spec_table = [
        [Paragraph("<b>Recommended Specialist:</b>", body_style), Paragraph(f"<b>{spec.get('recommended_specialist', 'Medical Specialist')}</b>", body_style)],
        [Paragraph("<b>Department:</b>", body_style), Paragraph(spec.get("department", "Clinical Diagnostic Imaging"), body_style)],
        [Paragraph("<b>Review Urgency:</b>", body_style), Paragraph(f"<b>{spec.get('review_priority', 'Standard Review')}</b>", body_style)],
        [Paragraph("<b>Action Protocol:</b>", body_style), Paragraph(spec.get("action_prompt", "Corroborate with specialist review."), body_style)],
    ]
    t_spec = Table(spec_table, colWidths=[140, 400])
    t_spec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 6))

    # 5. IMPORTANT NOTE
    story.append(Paragraph("5. IMPORTANT NOTE", section_heading))
    notice_text = (
        "TRUSTMED is an AI-assisted decision-support system designed to assist healthcare workflows. "
        "It is NOT an autonomous medical device and does NOT replace professional medical judgment, "
        "clinical diagnosis, or therapeutic advice of a qualified healthcare professional. All AI results "
        "must be interpreted by an appropriately qualified physician in clinical context."
    )
    story.append(Paragraph(notice_text, disclaimer_style))
    story.append(Spacer(1, 6))

    # 6. AI TECHNICAL DETAILS (SECONDARY AUDIT)
    story.append(Paragraph("6. AI TECHNICAL DETAILS (SECONDARY AUDIT)", ParagraphStyle(
        "SecHeadSub", parent=section_heading, fontSize=9.5, leading=13, textColor=colors.HexColor("#64748b")
    )))
    unc = analysis_result.get("uncertainty", {})
    ood = analysis_result.get("ood", {})
    stab = analysis_result.get("stability", {})
    q = analysis_result.get("quality", {})

    raw_conf_val = cal.get('raw_confidence', pred.get('top_probability', 0.0))
    cal_conf_val = cal.get('calibrated_confidence', cal_conf)

    tech_rows = [
        [Paragraph("<b>Indicator</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Details</b>", body_style)],
        [
            Paragraph("Image Quality", body_style),
            Paragraph(q.get("verdict", "GOOD"), body_style),
            Paragraph(f"Blur score: {q.get('blur_score')} (Laplacian variance)", body_style)
        ],
        [
            Paragraph("Calibrated Confidence", body_style),
            Paragraph(f"{round(float(cal_conf_val)*100, 1)}%", body_style),
            Paragraph(f"Raw: {round(float(raw_conf_val)*100, 1)}% | Temp T={cal.get('temperature', 1.0)}", body_style)
        ],
        [
            Paragraph("Uncertainty", body_style),
            Paragraph(unc.get("uncertainty_level", "LOW"), body_style),
            Paragraph(f"Variance: {unc.get('variance', 0.0)} ({unc.get('num_stochastic_passes', 10)} stochastic passes)", body_style)
        ],
        [
            Paragraph("Image Familiarity (OOD)", body_style),
            Paragraph(ood.get("image_familiarity", "LOW RISK"), body_style),
            Paragraph(f"Energy Score: {ood.get('energy_score', 0.0)}", body_style)
        ],
        [
            Paragraph("Result Stability", body_style),
            Paragraph(f"{stab.get('stability_percentage', 100.0)}%", body_style),
            Paragraph(stab.get("description", "Concordant"), body_style)
        ]
    ]
    t_tech = Table(tech_rows, colWidths=[130, 90, 320])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_tech)

    # Build PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
