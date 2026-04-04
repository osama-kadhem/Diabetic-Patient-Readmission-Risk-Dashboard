"""
src/week6_risk.py
-----------------
30-Day Diabetic Readmission Risk Predictor
Logistic Regression + Random Over-Sampling pipeline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MANIFEST_PATH = "clinical_models/week10_final/feature_manifest.json"

# Week 8 calibrated operating thresholds (Platt-scaled lr_classweight_w7)
# Best-F1 mode (default):      τ = 0.514
# High-recall / screening mode: τ = 0.604
THRESHOLD_F1     = 0.514
THRESHOLD_HR     = 0.604

# Human-readable labels for admission_type_id (UCI dataset coding)
ADMISSION_TYPE_LABELS: dict[str, str] = {
    "1": "1 — Emergency",
    "2": "2 — Urgent",
    "3": "3 — Elective",
    "4": "4 — Newborn",
    "5": "5 — Not Available",
    "6": "6 — NULL / Unknown",
    "7": "7 — Trauma Centre",
    "8": "8 — Not Mapped",
}

# Human-readable labels for discharge_disposition_id (UCI dataset coding)
DISCHARGE_LABELS: dict[str, str] = {
    "1":  "1 — Discharged to Home",
    "2":  "2 — Discharged to Short-Term Care",
    "3":  "3 — Discharged / Skilled Nursing Facility",
    "4":  "4 — Discharged / ICF",
    "5":  "5 — Discharged to Another Inpatient Care Facility",
    "6":  "6 — Discharged to Home with Health Service",
    "7":  "7 — Left AMA (Against Medical Advice)",
    "8":  "8 — Discharged / Home IV Provider",
    "9":  "9 — Admitted as Inpatient to This Hospital",
    "10": "10 — Neonate Discharged to Another Hospital",
    "11": "11 — Expired",
    "12": "12 — Still Patient / Expected to Return",
    "13": "13 — Hospice / Home",
    "14": "14 — Hospice / Medical Facility",
    "15": "15 — Discharged / Swing Bed",
    "16": "16 — Discharged / Outpatient Rehab",
    "17": "17 — Discharged / Psychiatric Hospital",
    "18": "18 — NULL / Unknown",
    "19": "19 — Discharged / Critical Access Hospital",
    "20": "20 — Discharged / Another Type of Health Care Institution",
    "22": "22 — Discharged / Rehab Facility",
    "23": "23 — Discharged / Long-Term Care Hospital",
    "24": "24 — Discharged / Nursing Facility — Medicaid",
    "25": "25 — Not Mapped",
    "27": "27 — Discharged / Federal Health Care Facility",
    "28": "28 — Discharged / Psychiatric Distinct Part Unit",
}

# Fields that use a label→value mapping selectbox
_CODED_FIELDS = {
    "admission_type_id":      ADMISSION_TYPE_LABELS,
    "discharge_disposition_id": DISCHARGE_LABELS,
}


def load_w6_manifest(manifest_path: str = MANIFEST_PATH) -> dict | None:
    """Load the feature manifest. Returns None on failure."""
    try:
        with open(manifest_path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        st.error(f"⚠️ ERR-404: Manifest not found at `{manifest_path}`.")
        return None
    except json.JSONDecodeError as exc:
        st.error(f"⚠️ ERR-JSON: Could not parse manifest — {exc}")
        return None


def _patch_lr_compat(pipeline) -> None:
    """Inject missing multi_class attribute for sklearn 1.6/1.8 compatibility."""
    try:
        from sklearn.linear_model import LogisticRegression
        for _, step in pipeline.steps:
            if isinstance(step, LogisticRegression) and not hasattr(step, "multi_class"):
                step.multi_class = "ovr"
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading readmission risk model…")
def load_w6_model(model_path: str):
    """Load and cache the risk pipeline. Returns None on failure."""
    try:
        p = Path(model_path)
        if not p.exists():
            st.error(f"⚠️ ERR-404: Model artifact not found at `{model_path}`.")
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipeline = joblib.load(p)
        _patch_lr_compat(pipeline)
        return pipeline
    except Exception as exc:
        st.error(f"⚠️ ERR-500: Failed to load model — {exc}")
        return None


def render_w6_form(manifest: dict, reset_key: int = 0) -> pd.DataFrame:
    """
    Render the patient input form from the manifest in a compact 3-column layout.
    """
    features  = manifest["features"]
    num_order = manifest["num_features"]
    cat_order = manifest["cat_features"]

    inputs: dict[str, object] = {}

    # Reorder numeric features based on Week 9 robustness findings
    # number_inpatient is highly sensitive (+0.036 delta), make it prominent
    primary_num = [f for f in num_order if f == "number_inpatient"]
    # lab procedures and outpatient are insensitive (+0.0003 delta), make them secondary
    secondary_num = [f for f in num_order if f in ["num_lab_procedures", "number_outpatient"]]
    # The rest are in the middle
    middle_num = [f for f in num_order if f not in primary_num and f not in secondary_num]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🚨 Primary Risk Driver**")
        for feat in primary_num:
            spec  = features[feat]
            step  = float(spec.get("step", 1))
            inputs[feat] = st.slider(
                label     = spec["description"],
                min_value = int(spec["min"]),
                max_value = int(spec["max"]),
                value     = int(spec["median"]),
                step      = int(step),
                key       = f"w6_slider_{feat}_{reset_key}",
            )
            
        st.markdown("**📊 Core Metrics**")
        for feat in middle_num + secondary_num:
            spec  = features[feat]
            step  = float(spec.get("step", 1))
            inputs[feat] = st.slider(
                label     = spec["description"],
                min_value = int(spec["min"]),
                max_value = int(spec["max"]),
                value     = int(spec["median"]),
                step      = int(step),
                key       = f"w6_slider_{feat}_{reset_key}",
            )

    # Split categorical features between col2 and col3
    mid = len(cat_order) // 2
    med_features = cat_order[:mid]
    adm_features = cat_order[mid:]

    with col2:
        st.markdown("**💊 Medications**")
        for feat in med_features:
            spec    = features[feat]
            allowed = spec["allowed"]
            
            if feat in _CODED_FIELDS:
                label_map   = _CODED_FIELDS[feat]
                label_list  = [label_map.get(v, v) for v in allowed]
                default_raw = spec.get("default", allowed[0])
                default_lbl = label_map.get(default_raw, default_raw)
                idx         = label_list.index(default_lbl) if default_lbl in label_list else 0

                selected_label = st.selectbox(
                    label   = spec["description"],
                    options = label_list,
                    index   = idx,
                    key     = f"w6_select_{feat}_{reset_key}"
                )
                reverse = {v: k for k, v in label_map.items()}
                inputs[feat] = reverse.get(selected_label, selected_label.split(" — ")[0])
            else:
                d   = spec.get("default", allowed[0])
                idx = allowed.index(d) if d in allowed else 0
                inputs[feat] = st.selectbox(
                    label   = spec["description"],
                    options = allowed,
                    index   = idx,
                    key     = f"w6_select_{feat}_{reset_key}"
                )

    with col3:
        st.markdown("**🏥 Admission Info**")
        for feat in adm_features:
            spec    = features[feat]
            allowed = spec["allowed"]
            
            if feat in _CODED_FIELDS:
                label_map   = _CODED_FIELDS[feat]
                label_list  = [label_map.get(v, v) for v in allowed]
                default_raw = spec.get("default", allowed[0])
                default_lbl = label_map.get(default_raw, default_raw)
                idx         = label_list.index(default_lbl) if default_lbl in label_list else 0

                selected_label = st.selectbox(
                    label   = spec["description"],
                    options = label_list,
                    index   = idx,
                    key     = f"w6_select_{feat}_{reset_key}"
                )
                reverse = {v: k for k, v in label_map.items()}
                inputs[feat] = reverse.get(selected_label, selected_label.split(" — ")[0])
            else:
                d   = spec.get("default", allowed[0])
                idx = allowed.index(d) if d in allowed else 0
                inputs[feat] = st.selectbox(
                    label   = spec["description"],
                    options = allowed,
                    index   = idx,
                    key     = f"w6_select_{feat}_{reset_key}"
                )

    ordered_cols = num_order + cat_order
    return pd.DataFrame([{k: inputs[k] for k in ordered_cols}])


def compute_risk_band(prob: float, mode: str = "best_f1") -> tuple[str, str]:
    """
    Map a calibrated probability to (band_label, css_colour).

    mode='best_f1' (default)  : τ_high=0.604, τ_mid=0.514
    mode='screening'          : same bands, but communicates high-recall intent
    """
    tau_high = THRESHOLD_HR
    tau_mid  = THRESHOLD_F1
    if prob >= tau_high:
        return "High", "#dc2626"
    if prob >= tau_mid:
        return "Moderate", "#d97706"
    return "Low", "#059669"


def interpret_risk(prob: float, band: str, mode: str = "best_f1") -> str:
    """Return a plain-English interpretation of the calibrated risk score."""
    pct = prob * 100
    mode_note = (
        "(screening mode — optimised for high recall)"
        if mode == "screening"
        else "(best-F1 operating point)"
    )
    if band == "High":
        return (
            f"⚠️ This patient has a **higher-than-average** calibrated readmission risk "
            f"({pct:.1f}%) {mode_note}. "
            "Early follow-up within **7 days** and targeted discharge planning are strongly recommended."
        )
    if band == "Moderate":
        return (
            f"🟡 This patient carries a **moderate** calibrated readmission risk "
            f"({pct:.1f}%) {mode_note}. "
            "Follow-up within **14 days** and outpatient care coordination are advised."
        )
    return (
        f"✅ This patient has a **lower-than-average** calibrated readmission risk "
        f"({pct:.1f}%) {mode_note}. "
        "Standard discharge protocols are appropriate."
    )
