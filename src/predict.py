import pandas as pd
import numpy as np

def predict_risk(df, pipeline, threshold_high=0.7, threshold_medium=0.4):
    df_pred = df.copy()

    feature_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]

    # keep only columns present in the uploaded data
    valid_cols = [c for c in feature_cols if c in df.columns]

    if not valid_cols:
        # no matching columns — return random scores as a fallback
        df_pred['risk_probability'] = np.random.uniform(0, 1, len(df))
    else:
        try:
            probs = pipeline.predict_proba(df[valid_cols])[:, 1]
            df_pred['risk_probability'] = probs
        except Exception as e:
            print(f"Prediction error: {e}")
            df_pred['risk_probability'] = np.random.uniform(0, 1, len(df))

    # assign risk band labels
    conditions = [
        (df_pred['risk_probability'] >= threshold_high),
        (df_pred['risk_probability'] >= threshold_medium) & (df_pred['risk_probability'] < threshold_high),
        (df_pred['risk_probability'] < threshold_medium)
    ]
    choices = ['High', 'Medium', 'Low']
    df_pred['risk_band'] = np.select(conditions, choices, default='Low')

    return df_pred


def rank_patients(df):
    # sort by descending risk and assign a priority rank
    df_ranked = df.sort_values('risk_probability', ascending=False).reset_index(drop=True)
    df_ranked['follow_up_priority'] = df_ranked.index + 1
    return df_ranked
