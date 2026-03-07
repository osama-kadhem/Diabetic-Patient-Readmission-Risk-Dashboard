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

MANIFEST_PATH = "clinical_models/feature_manifest.json"

# thresholds for risk banding
THRESHOLD_HIGH   = 0.40
THRESHOLD_MEDIUM = 0.20


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
    Render the patient input form from the manifest.
    Returns a single-row DataFrame ready for predict_proba().
    """
    features  = manifest["features"]
    num_order = manifest["num_features"]
    cat_order = manifest["cat_features"]

    inputs: dict[str, object] = {}

    col_left, col_right = st.columns(2)

    # numeric sliders
    with col_left:
        st.markdown("**📊 Clinical Measurements**")
        for feat in num_order:
            spec = features[feat]
            step = 1.0 if (spec["max"] - spec["min"]) > 5 else 0.1
            inputs[feat] = st.slider(
                label     = spec["description"],
                min_value = float(spec["min"]),
                max_value = float(spec["max"]),
                value     = float(spec["median"]),
                step      = step,
                key       = f"w6_slider_{feat}_{reset_key}",
                help      = f"Range: {spec['min']} – {spec['max']}  |  median: {spec['median']}"
            )

    # categorical dropdowns
    with col_right:
        st.markdown("**💊 Medications & Admission**")
        for feat in cat_order:
            spec    = features[feat]
            allowed = spec["allowed"]
            d       = spec.get("default", allowed[0])
            idx     = allowed.index(d) if d in allowed else 0
            inputs[feat] = st.selectbox(
                label   = spec["description"],
                options = allowed,
                index   = idx,
                key     = f"w6_select_{feat}_{reset_key}"
            )

    # return columns in the exact order the pipeline expects
    ordered_cols = num_order + cat_order
    return pd.DataFrame([{k: inputs[k] for k in ordered_cols}])


def compute_risk_band(prob: float) -> tuple[str, str]:
    """Map a probability to (band_label, css_colour)."""
    if prob >= THRESHOLD_HIGH:
        return "High", "#dc2626"
    if prob >= THRESHOLD_MEDIUM:
        return "Medium", "#d97706"
    return "Low", "#059669"


def interpret_risk(prob: float, band: str) -> str:
    """Return a plain-English interpretation of the predicted risk."""
    pct = prob * 100
    if band == "High":
        return (
            f"⚠️ This patient has a **higher-than-average** predicted risk of 30-day readmission "
            f"({pct:.1f}%). Early follow-up and targeted discharge planning are strongly recommended."
        )
    if band == "Medium":
        return (
            f"🟡 This patient carries a **moderate** predicted risk of 30-day readmission "
            f"({pct:.1f}%). Routine follow-up with care-coordination review is advised."
        )
    return (
        f"✅ This patient has a **lower-than-average** predicted risk of 30-day readmission "
        f"({pct:.1f}%). Standard discharge protocols appear appropriate."
    )
