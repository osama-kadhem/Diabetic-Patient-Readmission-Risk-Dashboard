import pandas as pd

# required columns
REQUIRED_COLUMNS = [
    'patient_nbr', 'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses'
]

def validate_csv(df):
    # check columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if len(missing_cols) > 0:
        return {
            'is_valid': False, 
            'errors': [f"Missing: {', '.join(missing_cols)}"]
        }
    
    return {'is_valid': True, 'errors': []}
