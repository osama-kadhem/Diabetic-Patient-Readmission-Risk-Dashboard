import pandas as pd

# Minimum columns the uploaded CSV must contain.
# Matches lr_classweight_w7_final (the deployed model): 8 numeric + 32 categorical.
REQUIRED_COLUMNS = [
    # Numeric features
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
    # Categorical features (core subset — sufficient for inference)
    'race', 'gender', 'age', 'admission_type_id', 'admission_source_id',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
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
