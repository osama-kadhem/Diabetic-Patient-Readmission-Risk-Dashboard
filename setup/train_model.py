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
    
    # 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the 80% used for training
    train_indices = X_train.index
    trained_data = df.loc[train_indices].copy()
    trained_data.to_csv('data/trained_data.csv', index=False)
    print(f"✅ Trained Data saved (80%): data/trained_data.csv ({len(trained_data)} records)")

    # Save the 20% used for testing (The "Demo" file)
    test_indices = X_test.index
    test_data = df.loc[test_indices].copy()
    test_data.to_csv('data/test_data.csv', index=False)
    print(f"✅ Test Data saved (20%): data/test_data.csv ({len(test_data)} records)")

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

