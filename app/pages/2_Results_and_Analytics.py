import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]  # points to app/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))



from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from vero_engine import VEROEngine
from pdf_report import PDFInputs, build_vero_pdf


# ----------------------------
# Safety
# ----------------------------
if sys.version_info >= (3, 12):
    st.error("Run this app with Python 3.11 (your .venv).")
    st.stop()

# FIXED: page_title keyword
st.set_page_config(page_title="VERO - Results & Visual Analytics", layout="wide")


# ----------------------------
# Paths
# ----------------------------
HERE = Path(__file__).resolve().parent.parent
BASE_MODEL = HERE / "vero_base_model_prefit.joblib"
CALIBRATOR = HERE / "vero_calibrator_prefit.joblib"
META = HERE / "vero_metadata.json"


# ----------------------------
# Engine loader
# ----------------------------
@st.cache_resource
def load_engine() -> VEROEngine:
    return VEROEngine(BASE_MODEL, CALIBRATOR, META)


# ----------------------------
# Helpers
# ----------------------------
def pretty_label(code: str) -> str:
    return str(code).replace("_", " ").strip().title()


def derive_age_group(age_value: Any) -> Optional[str]:
    if age_value is None:
        return None
    try:
        a = float(age_value)
    except Exception:
        return None
    return "<= 65 years" if a <= 65 else "> 65 years"


def clip_for_display(prob: float) -> float:
    eps = 1e-6
    p = float(prob)
    return float(min(max(p, eps), 1 - eps))


def phenotype_membership_label(prob: float, threshold: float = 0.5) -> str:
    return "Accelerated aging / frailty" if prob >= threshold else "Non-accelerated / lower frailty"


def risk_badge(stratum: str) -> str:
    color = {"Low": "#2e7d32", "Medium": "#ed6c02", "High": "#d32f2f"}.get(stratum, "#444")
    return f"""
    <div style="display:inline-block;padding:7px 14px;border-radius:999px;
                background:{color};color:white;font-weight:800;">
        {stratum} Risk
    </div>
    """


def safe_to_date(x: Any) -> Optional[date]:
    if x is None:
        return None
    if x is pd.NaT:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        return x.date()

    if isinstance(x, date):
        return x

    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None

def _norm_key(k: Any) -> str:
    return str(k).strip().lower().replace(" ", "_")

def get_date_anywhere(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    *candidate_keys: str
) -> Optional[date]:
    """
    Try multiple key variants across both dicts.
    - Matches exact keys
    - Also matches by normalized key (case/space insensitive)
    """
    if not isinstance(primary, dict):
        primary = {}
    if not isinstance(fallback, dict):
        fallback = {}

    # 1) direct lookup first (fast)
    for k in candidate_keys:
        if k in primary:
            d = safe_to_date(primary.get(k))
            if d is not None:
                return d
        if k in fallback:
            d = safe_to_date(fallback.get(k))
            if d is not None:
                return d

    # 2) normalized lookup
    primary_norm = {_norm_key(k): k for k in primary.keys()}
    fallback_norm = {_norm_key(k): k for k in fallback.keys()}

    for k in candidate_keys:
        nk = _norm_key(k)
        if nk in primary_norm:
            d = safe_to_date(primary.get(primary_norm[nk]))
            if d is not None:
                return d
        if nk in fallback_norm:
            d = safe_to_date(fallback.get(fallback_norm[nk]))
            if d is not None:
                return d

    return None

def fmt_prob(p: float) -> str:
    """
    Avoid ugly "0.0000" displays.
    """
    try:
        p = float(p)
    except Exception:
        return "-"

    if np.isnan(p):
        return "-"
    if p == 0.0:
        return "≈ 0 (very small)"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.6f}"


def make_membership_gauge(prob_display: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob_display),
            number={"valueformat": ".4f"},
            gauge={"axis": {"range": [0, 1]}},
            title={"text": "Membership probability (used for score)"},
        )
    )
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_timeline_figure(events: List[Dict[str, Any]]) -> go.Figure:
    """
    Timeline with staggered y positions to reduce label overlap.
    Uses month-year ticks for readability.
    """
    ev_df = pd.DataFrame(events).copy()
    ev_df = ev_df.dropna(subset=["date"]).sort_values("date")

    if ev_df.empty:
        return go.Figure()

    y = [(i % 2) for i in range(len(ev_df))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ev_df["date"],
            y=y,
            mode="markers+text",
            text=ev_df["event"],
            textposition="top center",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
        )
    )

    fig.add_shape(
        type="line",
        x0=min(ev_df["date"]),
        x1=max(ev_df["date"]),
        y0=0.5,
        y1=0.5,
        line=dict(width=2),
    )

    fig.update_yaxes(visible=False, range=[-0.6, 1.6])

    fig.update_xaxes(
        tickformat="%b %Y",
        tickangle=-25,
        showgrid=True,
        title="Date",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig


# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; max-width: 1200px;}
.stButton>button {border-radius: 12px; height: 44px; font-weight: 800;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Results & Visual Analytics")

with st.sidebar:
    st.markdown("### Utilities")
    if st.button("Reload engine (clear cache)", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ----------------------------
# Load engine + session input
# ----------------------------
engine = load_engine()
FEATURE_COLS = engine.feature_cols

if "patient_record" not in st.session_state:
    st.warning("No inputs found yet. Go to Patient Input first and save inputs.")
    st.stop()

patient: Dict[str, Any] = dict(st.session_state["patient_record"])
selected_id = st.session_state.get("selected_patient_id", None)

if "age_group" in FEATURE_COLS:
    patient["age_group"] = derive_age_group(patient.get("age"))

age_val = patient.get("age")
age_group_val = derive_age_group(age_val)
gender_val = patient.get("gender")
eth_val = patient.get("ethnicity")
edu_val = patient.get("education_level")
emp_val = patient.get("employment_status")

with st.expander("Patient summary (display only)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", selected_id or "(manual)")
    c2.metric("Gender", gender_val or "-")
    c3.metric("Age", "-" if age_val in [None, ""] else str(age_val))
    c4.metric("Age group", age_group_val or "-")

    demo_table = pd.DataFrame(
        [
            ("Ethnicity", eth_val or "-"),
            ("Education level", edu_val or "-"),
            ("Employment status", emp_val or "-"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(demo_table, use_container_width=True, hide_index=True)

st.divider()


# ----------------------------
# Compute button
# ----------------------------
compute = st.button("Compute VERO", type="primary", use_container_width=True)
if not compute:
    st.info("Click Compute VERO to generate score, contributors, timeline, and PDF.")
    st.stop()


# ----------------------------
# Predict
# ----------------------------
try:
    res = engine.predict_single(patient)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

p_base = float(res.base_probability)
p_cal = float(res.calibrated_probability)
p_used = float(res.probability_used_for_score)

score = int(res.vero_score)
stratum = str(res.risk_stratum)

p_used_disp = clip_for_display(p_used)


# ----------------------------
# Metrics (warning banner removed)
# ----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Probability used for score", fmt_prob(p_used))
m2.metric("Calibrated probability", fmt_prob(p_cal))
m3.metric("Base probability", fmt_prob(p_base))
m4.markdown(risk_badge(stratum), unsafe_allow_html=True)

st.divider()


# ----------------------------
# Membership panel
# ----------------------------
threshold = 0.5
membership_label = phenotype_membership_label(p_used, threshold=threshold)

left, right = st.columns([0.55, 0.45], gap="large")
with left:
    st.subheader("Phenotype membership")
    st.write(f"**Predicted membership:** {membership_label}")
    st.progress(p_used_disp)
    st.caption("Progress bar uses probability used for scoring (clipped only for UI).")

with right:
    gauge_fig = make_membership_gauge(p_used_disp)
    st.plotly_chart(gauge_fig, use_container_width=True)

st.divider()

# ----------------------------
# Timeline 
# ----------------------------
st.subheader("Patient timeline")

timeline_record = st.session_state.get("timeline_record", {}) or {}

# IMPORTANT: also search patient record as a fallback
timeline_source_primary = timeline_record
timeline_source_fallback = patient  # patient_record

timeline_fields = [
    # label, candidate keys (we try many variants)
    ("Observation start", ["observation_start_date", "Observation start date", "Observation Start Date"]),
    ("Observation end", ["observation_end_date", "Observation end date", "Observation End Date"]),
    ("Tumor diagnosis", ["tumor_diagnosis_date", "Tumor diagnosis date", "Diagnosis date", "diagnosis_date"]),
    ("Oncology unit intake", ["Oncology Unit Intake Date", "oncology_unit_intake_date", "Oncology intake date"]),
    ("Surgery", ["surgery_date", "Surgery date", "Surgery Date"]),
    ("Radiotherapy start", ["radiotherapy_start_date", "Radiotherapy start date", "Radiotherapy Start Date"]),
    ("Radiotherapy end", ["radiotherapy_end_date", "Radiotherapy end date", "Radiotherapy End Date"]),
]

events: List[Dict[str, Any]] = []
for label, keys in timeline_fields:
    d = get_date_anywhere(timeline_source_primary, timeline_source_fallback, *keys)
    if d is not None:
        events.append({"event": label, "date": d})

events.sort(key=lambda x: x["date"])

timeline_fig = None
if not events:
    st.info("No usable timeline dates found (nothing could be parsed).")
    # Optional debug (remove later if you want)
    with st.expander("Debug: timeline keys detected", expanded=False):
        st.write("timeline_record keys:", list(timeline_record.keys())[:80])
        st.write("patient_record keys:", list(patient.keys())[:80])
else:
    timeline_fig = make_timeline_figure(events)
    st.plotly_chart(timeline_fig, use_container_width=True)
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


# ----------------------------
# Contributors: TOP 10
# ----------------------------
TOP_K = 10
topk = engine.top_contributors_single(patient, top_k=TOP_K).copy()
topk["feature_code"] = topk["base_feature"]
topk["feature_label"] = topk["base_feature"].apply(pretty_label)
topk = topk[["feature_code", "feature_label", "total_contribution"]].reset_index(drop=True)

st.subheader(f"Top {TOP_K} contributors")
st.dataframe(topk, use_container_width=True, hide_index=True)

contrib_fig = px.bar(
    topk.sort_values("total_contribution"),
    x="total_contribution",
    y="feature_label",
    orientation="h",
    title=f"Top {TOP_K} contributors (signed contribution)",
)
contrib_fig.update_layout(height=max(420, 60 + 30 * len(topk)))
st.plotly_chart(contrib_fig, use_container_width=True)

st.divider()


# ----------------------------
# Export PDF
# ----------------------------
st.subheader("Export")
notes = st.text_area("Clinical notes (optional, for PDF)", value="", height=90)

try:
    contrib_png = pio.to_image(contrib_fig, format="png", scale=2)
    gauge_png = pio.to_image(gauge_fig, format="png", scale=2)
    timeline_png = pio.to_image(timeline_fig, format="png", scale=2) if timeline_fig is not None else b""
except Exception:
    st.error("Chart rendering for PDF failed. Install kaleido in your venv: pip install kaleido")
    contrib_png, gauge_png, timeline_png = b"", b"", b""

fields_provided = int(sum(1 for k in FEATURE_COLS if patient.get(k) not in [None, ""]))
fields_total = int(len(FEATURE_COLS))

pdf_inputs = PDFInputs(
    patient_id=selected_id or None,
    notes=notes.strip() or None,
    age_group=age_group_val,
    gender=gender_val,
    ethnicity=eth_val,
    education_level=edu_val,
    employment_status=emp_val,
    vero_probability=float(p_used),
    vero_probability_display=float(p_used_disp),
    vero_score=int(score),
    risk_stratum=stratum,
    fields_provided=fields_provided,
    fields_total=fields_total,
    phenotype_label=membership_label,
    phenotype_threshold=float(threshold),
)

pdf_bytes = build_vero_pdf(
    summary=pdf_inputs,
    top_contributors=topk,
    contrib_bar_png_bytes=contrib_png,
    gauge_png_bytes=gauge_png,
    timeline_png_bytes=timeline_png,
    timeline_events=events,
)

st.download_button(
    label="Download patient summary (PDF)",
    data=pdf_bytes,
    file_name=f"vero_summary_{selected_id or 'patient'}.pdf",
    mime="application/pdf",
    use_container_width=True,
)
