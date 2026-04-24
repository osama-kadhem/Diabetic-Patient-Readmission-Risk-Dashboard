import pytest
import pandas as pd
import numpy as np
from src.predict import predict_risk, rank_patients

class DummyPipeline:
    def __init__(self):
        self.feature_names_in_ = ['feature_a', 'feature_b']

    def predict_proba(self, X):
        # Return dummy probabilities
        # Return probability 0.8 for the first and 0.4 for the second
        return np.array([[0.2, 0.8], [0.6, 0.4]])


def test_predict_risk_leaky_columns_removed():
    df = pd.DataFrame({
        'feature_a': [1, 2],
        'feature_b': [3, 4],
        'readmitted': [1, 0], # Leaky column
        'target': [1, 0]      # Leaky column
    })
    
    pipeline = DummyPipeline()
    with pytest.warns(UserWarning, match="Data leakage detected"):
        result = predict_risk(df, pipeline)
        
    assert 'readmitted' not in result.columns
    assert 'target' not in result.columns
    assert 'risk_probability' in result.columns
    
def test_predict_risk_missing_all_columns():
    df = pd.DataFrame({
        'feature_c': [1, 2] # Missing a and b
    })
    
    pipeline = DummyPipeline()
    with pytest.raises(ValueError, match="REQUIRED FEATURES MISSING"):
        predict_risk(df, pipeline)


def test_predict_risk_bands():
    df = pd.DataFrame({
        'feature_a': [1, 2],
        'feature_b': [3, 4]
    })
    
    pipeline = DummyPipeline()
    result = predict_risk(df, pipeline, threshold_high=0.7, threshold_medium=0.5)
    
    # 0.8 should be High (>= 0.7)
    # 0.4 should be Low (< 0.5)
    assert result.loc[0, 'risk_band'] == 'High'
    assert result.loc[1, 'risk_band'] == 'Low'

def test_rank_patients():
    df = pd.DataFrame({
        'patient_id': ['A', 'B', 'C'],
        'risk_probability': [0.1, 0.9, 0.5]
    })
    
    ranked = rank_patients(df)
    
    assert ranked.loc[0, 'patient_id'] == 'B' # Highest risk
    assert ranked.loc[1, 'patient_id'] == 'C'
    assert ranked.loc[2, 'patient_id'] == 'A'
    
    assert ranked.loc[0, 'follow_up_priority'] == 1
    assert ranked.loc[1, 'follow_up_priority'] == 2
    assert ranked.loc[2, 'follow_up_priority'] == 3
