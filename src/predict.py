import pandas as pd
import numpy as np
import warnings

# The pipeline dictates the expected features at runtime.

# Columns that would constitute target leakage if present in the input
_LEAKY_COLUMNS = ['readmitted', 'readmitted_binary', 'label', 'target']

def predict_risk(df, pipeline, threshold_high=0.604, threshold_medium=0.514):
    df_pred = df.copy()

    # Strip target column if uploaded with the data
    leaky_found = [c for c in _LEAKY_COLUMNS if c in df_pred.columns]
    if leaky_found:
        warnings.warn(
            f"Data leakage detected: columns {leaky_found} were present in the "
            "uploaded file and have been removed before prediction. "
            "Predictions on labelled training data inflate apparent performance.",
            UserWarning, stacklevel=2
        )
        df_pred = df_pred.drop(columns=leaky_found)

    # Require ALL expected features
    try:
        expected_cols = list(pipeline.feature_names_in_)
    except AttributeError:
        raise RuntimeError("Pipeline must expose feature_names_in_. Ensure the model is fitted.")

    missing_cols = [c for c in expected_cols if c not in df_pred.columns]

    if len(missing_cols) == len(expected_cols):
        raise ValueError("REQUIRED FEATURES MISSING: None of the expected columns were found in the uploaded CSV.")
    elif missing_cols:
        warnings.warn(
            f"Missing features for prediction: {missing_cols}. "
            "Predictions may be unreliable if imputed.", UserWarning, stacklevel=2
        )
        # Attempt to use the existing columns; let sklearn handle/fail imputation if needed
        # But we must pass exactly expected_cols
        pass

    try:
        # Pipeline requires all expected training features
        probs = pipeline.predict_proba(df_pred[expected_cols])[:, 1]
        df_pred['risk_probability'] = probs
    except KeyError as e:
        raise RuntimeError(f"Prediction failed with missing columns. Error: {str(e)}") from e

    # Risk band assignment
    conditions = [
        (df_pred['risk_probability'] >= threshold_high),
        (df_pred['risk_probability'] >= threshold_medium) & (df_pred['risk_probability'] < threshold_high),
        (df_pred['risk_probability'] < threshold_medium)
    ]
    choices = ['High', 'Moderate', 'Low']
    df_pred['risk_band'] = np.select(conditions, choices, default='Low')

    return df_pred

def rank_patients(df):
    """Sort by descending risk and assign a 1-based priority rank."""
    df_ranked = df.sort_values('risk_probability', ascending=False).reset_index(drop=True)
    df_ranked['follow_up_priority'] = df_ranked.index + 1
    return df_ranked
