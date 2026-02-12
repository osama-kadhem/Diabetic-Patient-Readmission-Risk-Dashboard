import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
import os

# Features

NUMERIC_FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]


def load_real_data():
    # Load the real clinical dataset
    print("Loading real clinical data...")
    df = pd.read_csv('data/diabetic_data.csv')
    
    # Clean missing values ('?')
    df = df.replace('?', np.nan)
    
    # Target Encoding: readmitted <30 days = 1, otherwise = 0
    # Note: <30 is the high-risk class for readmission
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    return df

def create_and_save_pipeline():
    # train and save model
    print("Initializing Training Pipeline (Real Data)...")

    df = load_real_data()

    # Features used for the dashboard
    X = df[NUMERIC_FEATURES]
    y = df['readmitted']
    
    # 80/20 Patient-Level Split (Leakage-Safe)
    # We group by 'patient_nbr' so that if a patient has multiple encounters,
    # they all stay in either the train OR the test set, never both.
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['patient_nbr']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Save the 80% used for training
    trained_data = df.iloc[train_idx].copy()
    trained_data.to_csv('data/trained_data.csv', index=False)
    print(f"✅ Trained Data saved (80%): data/trained_data.csv ({len(trained_data)} records)")

    # Save the 20% used for testing (The "Demo" file)
    test_data = df.iloc[test_idx].copy()
    test_data.to_csv('data/test_data.csv', index=False)
    print(f"✅ Test Data saved (20%): data/test_data.csv ({len(test_data)} records)")
    
    # Verify zero patient overlap
    overlap = set(trained_data['patient_nbr']) & set(test_data['patient_nbr'])
    print(f"🛡️  Security Check: Patient Overlap is {len(overlap)}")

    # Tuning
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }
    
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])
    
    print("Optimizing model parameters...")
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    
    # Evaluation
    test_acc = grid_search.score(X_test, y_test)
    print(f"\nModel Performance on Unseen Data:")
    print(f" - Best C: {grid_search.best_params_['classifier__C']}")
    print(f" - F1 Score: {test_acc:.2%}")
    
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/pipeline.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    import hashlib
    with open('models/pipeline.pkl', 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open('models/pipeline.hash', 'w') as f:
        f.write(file_hash)
    
    print(f"Model saved and integrity hash generated.")
    return best_pipeline


if __name__ == '__main__':
    create_and_save_pipeline()

