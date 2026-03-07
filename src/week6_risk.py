"""
src/week6_risk.py
-----------------
Week 6 – 30-Day Diabetic Readmission Risk Predictor
Model: lr_ros_w6  (Logistic Regression + Random Over-Sampling)

Public API
----------
load_w6_manifest(manifest_path)        -> dict
load_w6_model(manifest)                -> pipeline | None
render_w6_form(manifest, reset_key)    -> pd.DataFrame   (single-row)
compute_risk_band(prob)                -> ("Low"|"Medium"|"High", color_css)
interpret_risk(prob, band)             -> str
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Manifest path (absorbed into clinical_models) ───────────────────────────
MANIFEST_PATH = "clinical_models/feature_manifest.json"

# ── Risk thresholds (aligned with rest of app) ───────────────────────────────
THRESHOLD_HIGH   = 0.40   # ≥40 % → High   (readmission base-rate ~11 %, calibrated upward)
THRESHOLD_MEDIUM = 0.20   # ≥20 % → Medium


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_w6_manifest(manifest_path: str = MANIFEST_PATH) -> dict | None:
    """Load and return the feature manifest as a dict. Returns None on failure."""
    try:
        with open(manifest_path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        st.error(
            f"⚠️ ERR-404: Manifest not found at `{manifest_path}`. "
            "Ensure `clinical_models/feature_manifest.json` exists."
        )
        return None
    except json.JSONDecodeError as exc:
        st.error(f"⚠️ ERR-JSON: Could not parse manifest — {exc}")
        return None


def _patch_lr_compat(pipeline) -> None:
    """
    Compatibility shim: models trained with sklearn ≥ 1.8.0 no longer store
    `multi_class` on LogisticRegression, but sklearn 1.6.x predict_proba still
    checks for it.  If the attribute is absent, inject it with the only value
    that makes sense for binary classifiers ('ovr').
    """
    try:
        from sklearn.linear_model import LogisticRegression
        for _, step in pipeline.steps:
            if isinstance(step, LogisticRegression) and not hasattr(step, "multi_class"):
                step.multi_class = "ovr"
    except Exception:
        pass  # Never crash the app over a compatibility patch


@st.cache_resource(show_spinner="Loading readmission risk model…")
def load_w6_model(model_path: str):
    """
    Load the readmission risk pipeline from the path stored in the manifest.
    Cached as a resource so it loads once per session.
    Returns the pipeline, or None on failure.
    """
    try:
        p = Path(model_path)
        if not p.exists():
            st.error(
                f"⚠️ ERR-404: Model artifact not found at `{model_path}`. "
                "Check that the model file is present in `clinical_models/`."
            )
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipeline = joblib.load(p)
        _patch_lr_compat(pipeline)
        return pipeline
    except Exception as exc:         # noqa: BLE001
        st.error(f"⚠️ ERR-500: Failed to load Week 6 model — {exc}")
        return None


def _get_defaults(manifest: dict) -> dict:
    """Build a dict of {feature: default_value} from the manifest."""
    defaults = {}
    for feat, spec in manifest["features"].items():
        if spec["type"] == "numeric":
            defaults[feat] = float(spec["median"])
        else:
            allowed = spec["allowed"]
            d = spec.get("default", allowed[0])
            defaults[feat] = d if d in allowed else allowed[0]
    return defaults


def render_w6_form(manifest: dict, reset_key: int = 0) -> pd.DataFrame:
    """
    Auto-render the patient input form from the manifest.

    Parameters
    ----------
    manifest  : dict  loaded from feature_manifest.json
    reset_key : int   increment this to reset all widgets to defaults

    Returns
    -------
    pd.DataFrame – single-row DataFrame with one column per feature,
                   in the exact order defined by the manifest pipeline.
    """
    features  = manifest["features"]
    num_order = manifest["num_features"]   # ordered list of numeric feature names
    cat_order = manifest["cat_features"]   # ordered list of categorical feature names

    inputs: dict[str, object] = {}

    col_left, col_right = st.columns(2)

    # ── Numeric (left column) ────────────────────────────────────────────────
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
                help      = f"Training range: {spec['min']} – {spec['max']}  |  median: {spec['median']}"
            )

    # ── Categorical (right column) ───────────────────────────────────────────
    with col_right:
        st.markdown("**💊 Medications & Admission**")
        for feat in cat_order:
            spec = features[feat]
            allowed = spec["allowed"]
            d      = spec.get("default", allowed[0])
            idx    = allowed.index(d) if d in allowed else 0
            inputs[feat] = st.selectbox(
                label   = spec["description"],
                options = allowed,
                index   = idx,
                key     = f"w6_select_{feat}_{reset_key}"
            )

    # Return in the canonical feature order (num_features + cat_features)
    ordered_cols = num_order + cat_order
    return pd.DataFrame([{k: inputs[k] for k in ordered_cols}])


def compute_risk_band(prob: float) -> tuple[str, str]:
    """
    Map a probability to (band_label, css_colour).

    Thresholds are purposely lower than the bulk-analysis tab because
    the manifested model (trained on resampled data) produces
    well-calibrated probabilities that cluster around 0.20 – 0.45 for
    truly high-risk patients.
    """
    if prob >= THRESHOLD_HIGH:
        return "High", "#dc2626"
    if prob >= THRESHOLD_MEDIUM:
        return "Medium", "#d97706"
    return "Low", "#059669"


def interpret_risk(prob: float, band: str) -> str:
    """Return a plain-English clinical interpretation string."""
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
