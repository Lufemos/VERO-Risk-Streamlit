from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)


# ----------------------------
# Formatting helpers
# ----------------------------
def _fmt_prob(p: float) -> str:
    """
    Human-friendly probability formatting.
    - Avoids showing 0.000 for tiny probabilities.
    - Keeps sensible decimals for normal values.
    """
    try:
        p = float(p)
    except Exception:
        return "NA"

    if pd.isna(p):
        return "NA"

    if p == 0.0:
        return "≈ 0 (very small)"

    if p < 0.001:
        return f"{p:.2e}"

    return f"{p:.6f}"


def _risk_color(stratum: str):
    if stratum == "Low":
        return colors.HexColor("#2e7d32")
    if stratum == "Medium":
        return colors.HexColor("#ed6c02")
    if stratum == "High":
        return colors.HexColor("#d32f2f")
    return colors.HexColor("#444444")


def _safe_img(png_bytes: bytes, width_cm: float, height_cm: float) -> Optional[Image]:
    if not png_bytes:
        return None
    img_buf = BytesIO(png_bytes)
    img = Image(img_buf)
    img.drawWidth = width_cm * cm
    img.drawHeight = height_cm * cm
    return img


def _kv_table(rows: List[List[str]], col_widths_cm: List[float]) -> Table:
    t = Table(rows, colWidths=[w * cm for w in col_widths_cm], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _styled_table(
    data: List[List[Any]],
    col_widths_cm: List[float],
    header_bg: str = "#f2f2f2",
    font_size: int = 9,
    repeat_rows: int = 1,
) -> Table:
    t = Table(data, colWidths=[w * cm for w in col_widths_cm], hAlign="LEFT", repeatRows=repeat_rows)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


# ----------------------------
# Data structure
# ----------------------------
@dataclass
class PDFInputs:
    patient_id: Optional[str]
    notes: Optional[str]

    age_group: Optional[str]
    gender: Optional[str]
    ethnicity: Optional[str]
    education_level: Optional[str]
    employment_status: Optional[str]

    vero_probability: float
    vero_probability_display: float
    vero_score: int
    risk_stratum: str
    fields_provided: int
    fields_total: int
    phenotype_label: str
    phenotype_threshold: float


# ----------------------------
# PDF builder
# ----------------------------
def build_vero_pdf(
    summary: PDFInputs,
    top_contributors: pd.DataFrame,
    contrib_bar_png_bytes: bytes,
    gauge_png_bytes: bytes,
    timeline_png_bytes: bytes,
    timeline_events: List[Dict[str, Any]],
) -> bytes:
    """
    Returns PDF bytes.
    """
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="VERO Patient Summary",
        author="VERO Risk Calculator",
    )

    styles = getSampleStyleSheet()

    # Slightly nicer normal style
    small_grey = ParagraphStyle(
        "small_grey",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#666666"),
    )

    story: List[Any] = []

    # Title
    story.append(Paragraph("<b>VERO Patient Summary</b>", styles["Title"]))
    story.append(Spacer(1, 0.10 * cm))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.35 * cm))

    # ----------------------------
    # Patient details
    # ----------------------------
    pid = summary.patient_id or "Not provided"
    story.append(Paragraph("<b>Patient details</b>", styles["Heading2"]))

    demo_rows = [
        ["Field", "Value"],
        ["Patient ID", pid],
        ["Age group", summary.age_group or "Unknown"],
        ["Gender", summary.gender or "Unknown"],
        ["Ethnicity", summary.ethnicity or "Unknown"],
        ["Education level", summary.education_level or "Unknown"],
        ["Employment status", summary.employment_status or "Unknown"],
    ]
    story.append(_kv_table(demo_rows, col_widths_cm=[5.2, 11.0]))
    story.append(Spacer(1, 0.35 * cm))

    # ----------------------------
    # Risk summary
    # ----------------------------
    story.append(Paragraph("<b>Risk summary</b>", styles["Heading2"]))
    risk_col = _risk_color(summary.risk_stratum)

    # IMPORTANT: use _fmt_prob() here (no more 0.000)
    risk_rows = [
        ["Metric", "Value"],
        ["Probability used for score", _fmt_prob(summary.vero_probability_display)],
        ["VERO score (0-100)", str(int(summary.vero_score))],
        ["Risk stratum", str(summary.risk_stratum)],
        ["Fields provided", f"{int(summary.fields_provided)}/{int(summary.fields_total)}"],
    ]
    score_table = Table(risk_rows, colWidths=[7.2 * cm, 9.0 * cm], hAlign="LEFT")
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # Highlight risk row
                ("BACKGROUND", (0, 3), (-1, 3), risk_col),
                ("TEXTCOLOR", (0, 3), (-1, 3), colors.white),
            ]
        )
    )
    story.append(score_table)
    story.append(Spacer(1, 0.35 * cm))

    # ----------------------------
    # Phenotype membership
    # ----------------------------
    story.append(Paragraph("<b>Phenotype membership</b>", styles["Heading2"]))
    story.append(Paragraph(f"Predicted membership: <b>{summary.phenotype_label}</b>", styles["Normal"]))
    story.append(Paragraph(f"Threshold used: {float(summary.phenotype_threshold):.2f}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * cm))

    gauge_img = _safe_img(gauge_png_bytes, width_cm=16.0, height_cm=5.6)
    if gauge_img is not None:
        story.append(gauge_img)
        story.append(Spacer(1, 0.30 * cm))
    else:
        story.append(Paragraph("Gauge not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    # ----------------------------
    # Contributors
    # ----------------------------
    story.append(Paragraph("<b>Top contributors</b>", styles["Heading2"]))

    tdf = top_contributors.copy()
    cols = ["feature_code", "feature_label", "total_contribution"]
    for c in cols:
        if c not in tdf.columns:
            raise ValueError(f"Missing column in top_contributors: {c}")

    tdf["total_contribution"] = pd.to_numeric(tdf["total_contribution"], errors="coerce").fillna(0.0)
    tdf["total_contribution"] = tdf["total_contribution"].round(4)

    table_data = [cols] + tdf[cols].values.tolist()

    # repeatRows=1 is inside _styled_table so header repeats on new pages
    contrib_table = _styled_table(
        data=table_data,
        col_widths_cm=[5.0, 7.0, 3.4],
        font_size=9,
        repeat_rows=1,
    )
    story.append(contrib_table)
    story.append(Spacer(1, 0.30 * cm))

    story.append(Paragraph("<b>Contributor chart</b>", styles["Heading2"]))
    contrib_img = _safe_img(contrib_bar_png_bytes, width_cm=16.0, height_cm=6.8)
    if contrib_img is not None:
        story.append(contrib_img)
        story.append(Spacer(1, 0.30 * cm))
    else:
        story.append(Paragraph("Contributor chart not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    # ----------------------------
    # Timeline
    # ----------------------------
    story.append(Paragraph("<b>Patient timeline</b>", styles["Heading2"]))

    timeline_img = _safe_img(timeline_png_bytes, width_cm=16.0, height_cm=5.8)
    if timeline_img is not None:
        story.append(timeline_img)
        story.append(Spacer(1, 0.20 * cm))
    else:
        story.append(Paragraph("Timeline chart not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    if timeline_events:
        ev_df = pd.DataFrame(timeline_events).copy()
        if "date" in ev_df.columns:
            ev_df["date_str"] = ev_df["date"].astype(str)
        else:
            ev_df["date_str"] = ""

        if "event" not in ev_df.columns:
            ev_df["event"] = ""

        ev_df = ev_df[["event", "date_str"]].copy()

        tdata = [["Event", "Date"]] + ev_df.values.tolist()

        ttable = _styled_table(
            data=tdata,
            col_widths_cm=[9.0, 6.6],
            font_size=9,
            repeat_rows=1,
        )
        story.append(ttable)
    else:
        story.append(Paragraph("No timeline events provided.", styles["Normal"]))

    story.append(Spacer(1, 0.35 * cm))

    # ----------------------------
    # Notes + disclaimer
    # ----------------------------
    story.append(Paragraph("<b>Clinical notes</b>", styles["Heading2"]))
    story.append(Paragraph(summary.notes or "None", styles["Normal"]))

    story.append(Spacer(1, 0.55 * cm))
    story.append(
        Paragraph(
            "Decision support only. Interpret alongside clinical judgment.",
            small_grey,
        )
    )

    doc.build(story)
    return buf.getvalue()
