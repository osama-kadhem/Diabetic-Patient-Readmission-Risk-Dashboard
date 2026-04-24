import pytest
import pandas as pd
from src.data_validation import validate_csv, REQUIRED_COLUMNS

def test_validate_csv_success():
    # Create dataframe with all required columns
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df.loc[0] = [0] * len(REQUIRED_COLUMNS)
    
    result = validate_csv(df)
    assert result['is_valid'] is True
    assert len(result['errors']) == 0

def test_validate_csv_missing_columns():
    # Create dataframe missing first 3 required columns
    df = pd.DataFrame(columns=REQUIRED_COLUMNS[3:])
    
    result = validate_csv(df)
    assert result['is_valid'] is False
    assert len(result['errors']) == 1
    assert "Missing:" in result['errors'][0]
    assert REQUIRED_COLUMNS[0] in result['errors'][0]
    assert REQUIRED_COLUMNS[1] in result['errors'][0]
    assert REQUIRED_COLUMNS[2] in result['errors'][0]
