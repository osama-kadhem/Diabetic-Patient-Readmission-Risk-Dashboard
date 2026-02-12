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

def create_sample_data(n_samples=5000):
    # generate mock data

    np.random.seed(42)
    
    # Categorical distributions
    races = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other']
    genders = ['Female', 'Male']
    age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    
    data = {
        'patient_nbr': [f'PN{100000+i}' for i in range(n_samples)],
        'race': np.random.choice(races, n_samples, p=[0.7, 0.2, 0.05, 0.02, 0.03]),
        'gender': np.random.choice(genders, n_samples),
        'age': np.random.choice(age_groups, n_samples, p=[0.01, 0.01, 0.02, 0.03, 0.1, 0.15, 0.25, 0.25, 0.15, 0.03]),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_procedures': np.random.randint(0, 7, n_samples),
        'num_medications': np.random.randint(1, 50, n_samples),
        'number_outpatient': np.random.randint(0, 10, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 10, n_samples),
        'number_diagnoses': np.random.randint(1, 16, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # simulation logic

    base_risk = (
        df['time_in_hospital'] / 14 * 0.2 +
        df['number_inpatient'] / 10 * 0.5 +
        df['num_medications'] / 50 * 0.15 +
        df['number_diagnoses'] / 16 * 0.15
    )
    
    # Binary readmission (1 if risk high)
    df['readmitted'] = (base_risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return df

def create_and_save_pipeline():
    # train and save model

    print("Training...")

    df = create_sample_data(10000)

    
    X = df[NUMERIC_FEATURES]
    y = df['readmitted']
    
    # Split: Train (60%), Validation (20%), Test (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
    
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Tuning

    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2']
    }
    
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])
    
    print("Optimizing...")

    grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print(f"Optimal Regularization found: C={grid_search.best_params_['classifier__C']}")
    
    # Evaluation
    train_acc = grid_search.score(X_train, y_train)
    test_acc = grid_search.score(X_test, y_test)
    
    print(f"\nFinal Stats:")
    print(f" - Accuracy: {test_acc:.2%}")
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/pipeline.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    import hashlib
    with open('models/pipeline.pkl', 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open('models/pipeline.hash', 'w') as f:
        f.write(file_hash)
    
    print(f"\nModel saved and integrity hash generated.")

    return best_pipeline

def save_sample_csv(df, n=100):
    path = 'data/diabetes_test_data.csv'
    df.head(n).to_csv(path, index=False)
    print(f"✅ Sample CSV saved: {path}")

if __name__ == '__main__':
    create_and_save_pipeline()
    df = create_sample_data(100)
    save_sample_csv(df)

