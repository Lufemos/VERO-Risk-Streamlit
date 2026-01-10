from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import numpy as np
import pandas as pd
import streamlit as st

from vero_engine import VEROEngine


# =============================================================================
# Helpers: session-state auto fill
# =============================================================================
LEAVE_BLANK_LABEL = "(leave blank)"
MISSING_LABELS = {"Not Known / Missing", "Missing / Not Noted"}


def _clean_value(v: Any) -> Any:
    """Convert NaN/NaT to empty string; keep normal python types."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, np.generic):
        return v.item()
    return v


def apply_patient_to_session(row: pd.Series, feature_cols: List[str]) -> None:
    """
    Push patient row values into st.session_state using *feature column names* as keys.
    Then rerun so widgets read the updated state.
    """
    for col in feature_cols:
        st.session_state[col] = _clean_value(row.get(col, ""))

    st.session_state["_active_patient_id"] = _clean_value(row.get("patient_id", ""))
    st.rerun()


def clean_levels(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if v is None:
            continue
        v = str(v).strip()
        if v == "" or v in MISSING_LABELS:
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def build_levels_from_df(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, List[str]]:
    levels: Dict[str, List[str]] = {}
    for c in cat_cols:
        if c in df.columns:
            vals = df[c].dropna().tolist()
            levels[c] = clean_levels(vals)
    return levels


def derive_age_group(age_value: Any) -> Optional[str]:
    if age_value is None or age_value == "":
        return None
    try:
        a = float(age_value)
    except Exception:
        return None
    return "<= 65 years" if a <= 65 else "> 65 years"


def pretty_label(code: str) -> str:
    return str(code).replace("_", " ").strip().title()


def parse_numeric(raw: str) -> Optional[float]:
    raw = str(raw).strip()
    if raw == "":
        return None
    try:
        return float(raw) if "." in raw else int(raw)
    except Exception:
        return None


def set_default_state_if_missing(col: str, default_value: Any) -> None:
    """
    Ensure st.session_state[col] exists before widget renders.
    This is important because widgets pull from session_state once a key is set.
    """
    if col not in st.session_state:
        st.session_state[col] = _clean_value(default_value)


def input_widget(
    col: str,
    numeric_cols: set,
    levels: Dict[str, List[str]],
    default_value: Any,
) -> Any:
    """
    Widget keys must be exactly `col` to allow session-state autofill.
    """
    label = pretty_label(col)
    set_default_state_if_missing(col, default_value)

    # numeric -> text_input (lets you leave blank)
    if col in numeric_cols:
        raw_default = st.session_state.get(col, "")
        raw = st.text_input(
            label,
            key=col,
            value="" if raw_default is None else str(raw_default),
            placeholder="Leave blank if unknown",
        )
        v = parse_numeric(raw)
        if str(raw).strip() != "" and v is None:
            st.warning(f"{label} expects a number. Leave blank if unknown.")
        return v

    # categorical dropdown
    if col in levels and len(levels[col]) > 0:
        opts = [LEAVE_BLANK_LABEL] + levels[col]

        current = st.session_state.get(col, "")
        current = "" if current is None else str(current).strip()

        if current == "":
            idx = 0
        else:
            idx = opts.index(current) if current in opts else 0

        pick = st.selectbox(label, options=opts, index=idx, key=col)
        return None if pick == LEAVE_BLANK_LABEL else pick

    # fallback free text
    raw_default = st.session_state.get(col, "")
    raw = st.text_input(
        label,
        key=col,
        value="" if raw_default is None else str(raw_default),
        placeholder="Type or leave blank if unknown",
    )
    return None if str(raw).strip() == "" else str(raw).strip()


# =============================================================================
# Paths / config
# =============================================================================
APP_DIR = Path(__file__).resolve().parents[1]  # app/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if sys.version_info < (3, 10):
    st.error("Please use Python 3.10+")
    st.stop()

st.set_page_config(page_title="VERO - Patient Input", layout="wide")

HERE = Path(__file__).resolve().parent.parent  # app/
BASE_MODEL = HERE / "vero_base_model_prefit.joblib"
CALIBRATOR = HERE / "vero_calibrator_prefit.joblib"
META = HERE / "vero_metadata.json"
DEFAULT_DATA_PATH = HERE / "data" / "codige_master_clean__v2.xlsx"


@st.cache_resource
def load_engine() -> VEROEngine:
    return VEROEngine(BASE_MODEL, CALIBRATOR, META)


@st.cache_data
def load_master_df_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =============================================================================
# Layout definitions
# =============================================================================
CATEGORIES: Dict[str, List[str]] = {
    "Demographics & Socio-economic": [
        "age", "gender", "ethnicity", "education_level",
        "employment_status", "alcohol_consumption", "smoking_status_detail",
    ],
    "Tumor & Molecular Context": [
        "tumor_type", "molecular_alterations", "mutations_present", "genotipo_DPYD_type",
    ],
    "Treatment Exposure (Baseline)": [
        "surgical_intervention", "radiotherapy_status", "received_chemo", "received_targeted_therapy",
        "oncology_treatment_lines_n", "n_treatment_lines", "max_combo_regimen_size",
        "total_chemo_cycles", "treatment_duration_days",
    ],
    "Comorbidities & Clinical Conditions": [
        "hypertension", "dyslipidemia", "ischemic_heart_disease", "atrial_fibrillation",
        "hypertensive_heart_disease", "diabete_tipo_II", "obesity_comorbidity", "copd", "asthma",
        "renal_insufficiency", "anemia_comorbidity", "depressive_syndrome", "psychiatric_disorders",
        "cerebrovascular_disorders", "gastroesophageal_reflux_full", "gastrointestinal_disorders",
        "cardiovascular_disorders",
    ],
    "Frailty & Burden Indices": [
        "cci_score", "IPB", "farmaci_cat_n", "total_unique_active_drugs",
    ],
    "Laboratory Ranges": [
        "white_blood_cells_range", "red_blood_cells_range", "hemoglobin_range",
        "neutrophils_percent_range", "platelet_count_range", "creatinine_range",
        "ast_got_range", "alt_gpt_range", "total_bilirubin_range", "direct_bilirubin_range",
    ],
    "ADR Summary": [
        "adr_n_tot",
    ],
}


# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; max-width: 1200px;}
h1 {letter-spacing: 0.2px;}
.section-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: rgba(0,0,0,0.72);
  margin-top: 0.4rem;
}
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px 6px 14px;
  background: rgba(255,255,255,0.8);
}
.card h3 {
  margin: 0 0 10px 0;
  font-size: 1.0rem;
}
.small-note {color: rgba(0,0,0,0.6); font-size: 0.9rem;}
hr {margin: 1rem 0;}
.stButton>button {border-radius: 12px; height: 44px; font-weight: 700;}
</style>
""",
    unsafe_allow_html=True
)


# =============================================================================
# Main
# =============================================================================
st.title("Patient Input")

engine = load_engine()
FEATURE_COLS = engine.feature_cols
NUM_COLS = set(engine.meta.get("numeric_cols", []))
CAT_COLS = list(engine.meta.get("categorical_cols", []))

# ----------------------------
# Data source
# ----------------------------
st.markdown('<div class="section-title">Data source</div>', unsafe_allow_html=True)
left_src, right_src = st.columns([0.65, 0.35], gap="large")

with left_src:
    up = st.file_uploader("Upload master Excel (optional)", type=["xlsx"])

with right_src:
    st.caption("If you do not upload, the app will use:")
    st.code(str(DEFAULT_DATA_PATH), language="text")

df_master: Optional[pd.DataFrame] = None
if up is not None:
    st.session_state["uploaded_master_bytes"] = up.getvalue()
    df_master = pd.read_excel(BytesIO(st.session_state["uploaded_master_bytes"]))
    df_master.columns = [str(c).strip() for c in df_master.columns]
    st.success("Uploaded master file loaded into session.")
else:
    if DEFAULT_DATA_PATH.exists():
        df_master = load_master_df_from_path(DEFAULT_DATA_PATH)

if df_master is None:
    st.warning("No master dataset loaded yet. Upload the Excel file to enable patient selection.")
    st.stop()

if "patient_id" not in df_master.columns:
    st.error("patient_id column not found in the master dataset.")
    st.stop()

levels = build_levels_from_df(df_master, CAT_COLS)

st.divider()

# ----------------------------
# Patient selection (auto-fill)
# ----------------------------
st.markdown('<div class="section-title">Select patient (auto-fill)</div>', unsafe_allow_html=True)

ids = (
    df_master["patient_id"]
    .astype(str).str.strip()
    .dropna()
    .unique()
    .tolist()
)
ids = sorted(ids)

selected = st.selectbox(
    "patient_id",
    options=["(none)"] + ids,
    index=0,
    key="patient_id_select",
)

# Apply patient only when user changes selection
if selected != "(none)":
    active = st.session_state.get("_active_patient_id", None)
    if active != selected:
        row_df = df_master.loc[df_master["patient_id"].astype(str).str.strip() == selected]
        if row_df.empty:
            st.warning("Selected patient_id not found in master data.")
        else:
            st.success(f"Loaded patient record for: {selected}")
            apply_patient_to_session(row_df.iloc[0], FEATURE_COLS)

# timeline for reporting (store, even if none)
timeline_record: Dict[str, Any] = {}
if selected != "(none)":
    row = df_master.loc[df_master["patient_id"].astype(str).str.strip() == selected].iloc[0]
    timeline_record = {
        "Diagnosis date": row.get("tumor_diagnosis_date", None),
        "Treatment start date": row.get("Oncology Unit Intake Date", row.get("observation_start_date", None)),
        "Radiotherapy start date": row.get("radiotherapy_start_date", None),
        "Chemo start date": None,
        "Targeted therapy start date": None,
        "Progression date": None,
        "Last follow-up date": row.get("observation_end_date", None),
    }

st.session_state["timeline_record"] = dict(timeline_record)

st.divider()

# ----------------------------
# Manual entry form (grouped)
# ----------------------------
st.markdown('<div class="section-title">Inputs (grouped)</div>', unsafe_allow_html=True)
st.caption("Leave any field blank if unknown. The model imputers handle missingness.")

with st.form("patient_form", clear_on_submit=False):
    st.markdown(
        '<div class="small-note">Auto-filled values (if any) are already loaded into the fields below.</div>',
        unsafe_allow_html=True
    )

    # render widgets using keys == feature names
    for category, cols in CATEGORIES.items():
        st.markdown(f'<div class="card"><h3>{category}</h3>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 1], gap="large")
        buckets = [c1, c2, c3]
        bi = 0

        for col in cols:
            if col not in FEATURE_COLS:
                continue
            if col == "age_group":
                continue

            with buckets[bi % 3]:
                _ = input_widget(
                    col=col,
                    numeric_cols=NUM_COLS,
                    levels=levels,
                    default_value=st.session_state.get(col, ""),
                )
            bi += 1

        st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("Save inputs for scoring", type="primary", use_container_width=True)

if submit:
    merged: Dict[str, Any] = {}

    # Pull from session_state (these are the actual widget values)
    for c in FEATURE_COLS:
        if c == "age_group":
            continue
        merged[c] = st.session_state.get(c, None)
        if merged[c] == "":
            merged[c] = None

    # Derived field for model
    if "age_group" in FEATURE_COLS:
        merged["age_group"] = derive_age_group(merged.get("age"))

    # Ensure all features exist
    for c in FEATURE_COLS:
        merged.setdefault(c, None)

    st.session_state["patient_record"] = merged

    st.session_state["selected_patient_id"] = None if selected == "(none)" else selected

    st.session_state["display_demographics"] = {
        "patient_id": st.session_state.get("selected_patient_id", None),
        "age": merged.get("age"),
        "age_group": merged.get("age_group"),
        "gender": merged.get("gender"),
        "ethnicity": merged.get("ethnicity"),
        "education_level": merged.get("education_level"),
        "employment_status": merged.get("employment_status"),
    }

    st.success("Saved. Go to Results & Visual Analytics to compute.")
