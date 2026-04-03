import pandas as pd
import numpy as np
import warnings

# These are the ONLY features the Weeks 4/5 pipeline was trained on.
# Do NOT add target-adjacent columns (e.g. discharge_disposition_id=11
# means the patient died — a direct proxy for non-readmission).
FEATURE_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

# Columns that would constitute target leakage if present in the input
_LEAKY_COLUMNS = ['readmitted', 'readmitted_binary', 'label', 'target']

def predict_risk(df, pipeline, threshold_high=0.7, threshold_medium=0.4):
    df_pred = df.copy()

    # ── Leakage guard 1: strip target column if uploaded with the data ────────
    leaky_found = [c for c in _LEAKY_COLUMNS if c in df_pred.columns]
    if leaky_found:
        warnings.warn(
            f"Data leakage detected: columns {leaky_found} were present in the "
            "uploaded file and have been removed before prediction. "
            "Predictions on labelled training data inflate apparent performance.",
            UserWarning, stacklevel=2
        )
        df_pred = df_pred.drop(columns=leaky_found)

    # ── Leakage guard 2: require ALL expected features ─────────────────────────
    # Predicting with a partial feature set would silently pass wrong-shape data
    # to the scaler, producing unreliable probabilities.
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    valid_cols   = [c for c in FEATURE_COLS if c in df.columns]

    if not valid_cols:
        # No matching features at all
        raise ValueError(
            f"REQUIRED FEATURES MISSING: None of the expected columns {FEATURE_COLS} "
            "were found in the uploaded CSV. Pipeline cannot generate predictions."
        )
    elif missing_cols:
        # Partial feature set — still warn, but attempt prediction
        warnings.warn(
            f"Missing features for prediction: {missing_cols}. "
            "Predictions may be unreliable. Ensure the uploaded CSV contains all "
            "required columns.",
            UserWarning, stacklevel=2
        )
        try:
            # Pipeline requires all 8 training features; predict on the full expected set.
            probs = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
            df_pred['risk_probability'] = probs
        except Exception as e:
            raise RuntimeError(
                f"Prediction failed with missing columns {missing_cols}. "
                f"Error: {str(e)}. Please ensure your CSV contains all 8 required features."
            ) from e
    else:
        # All features present — clean prediction path
        try:
            probs = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
            df_pred['risk_probability'] = probs
        except Exception as e:
            raise RuntimeError(
                f"predict_proba failed: {type(e).__name__}: {e}. "
                f"This is often due to a stale cache or version mismatch. "
                f"Click 'Clear Model Cache' in the sidebar and re-run analysis."
            ) from e

    # ── Risk band assignment ──────────────────────────────────────────────────
    conditions = [
        (df_pred['risk_probability'] >= threshold_high),
        (df_pred['risk_probability'] >= threshold_medium) & (df_pred['risk_probability'] < threshold_high),
        (df_pred['risk_probability'] < threshold_medium)
    ]
    choices = ['High', 'Medium', 'Low']
    df_pred['risk_band'] = np.select(conditions, choices, default='Low')

    return df_pred

def rank_patients(df):
    """Sort by descending risk and assign a 1-based priority rank."""
    df_ranked = df.sort_values('risk_probability', ascending=False).reset_index(drop=True)
    df_ranked['follow_up_priority'] = df_ranked.index + 1
    return df_ranked
