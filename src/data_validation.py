import pandas as pd

# Minimum columns the uploaded CSV must contain.
# Matches the final model features defined in feature_manifest.json.
REQUIRED_COLUMNS = [
    # Numeric features
    'time_in_hospital', 'num_medications',
    'number_inpatient', 'number_emergency', 'number_outpatient',
    # Categorical features
    'age', 'A1Cresult', 'insulin', 'change', 'diabetesMed',
    'metformin', 'glipizide', 'glyburide', 'max_glu_serum',
    'admission_type_id', 'discharge_disposition_id',
]

def validate_csv(df):
    """Check that the uploaded DataFrame contains every required column."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        return {
            'is_valid': False,
            'errors': [f"Missing: {', '.join(missing_cols)}"]
        }

    return {'is_valid': True, 'errors': []}
