"""
Week 4: Train imbalance strategy models compatible with the app environment
This script trains 3 models using the same features as the app expects
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, brier_score_loss
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Features matching the app
FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]

def load_data():
    """Load and prepare the training data"""
    print("Loading data...")
    df = pd.read_csv('data/trained_data.csv')
    
    X = df[FEATURES]
    y = df['readmitted']
    
    # Patient-level split to avoid leakage
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['patient_nbr']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive rate: {y_train.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, version_id):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    metrics = {
        "version_id": version_id,
        "PR_AUC": float(pr_auc),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "Brier": float(brier_score_loss(y_test, y_proba)),
        "threshold": 0.5
    }
    
    return metrics

def train_baseline(X_train, y_train):
    """Train baseline model (no imbalance handling)"""
    print("\n=== Training Baseline Model ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_classweight(X_train, y_train):
    """Train model with class weights"""
    print("\n=== Training Class Weight Model ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_ros(X_train, y_train):
    """Train model with Random Over Sampling"""
    print("\n=== Training ROS Model ===")
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', RandomOverSampler(random_state=42)),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create output directory
    output_dir = Path('clinical_models')
    output_dir.mkdir(exist_ok=True)
    
    # Train models
    models = {
        'baseline_v1': (train_baseline(X_train, y_train), "No imbalance handling"),
        'classweight_v1': (train_classweight(X_train, y_train), "class_weight='balanced'"),
        'ros_v1': (train_ros(X_train, y_train), "Random Over Sampling")
    }
    
    # Save models and metrics
    print("\n=== Saving Models ===")
    for version_id, (model, notes) in models.items():
        # Save model
        model_path = output_dir / f"{version_id}.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_path}")
        
        # Evaluate and save metrics
        metrics = evaluate_model(model, X_test, y_test, version_id)
        metrics['notes'] = notes
        
        json_path = output_dir / f"{version_id}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved {json_path}")
        
        # Print metrics
        print(f"   PR-AUC: {metrics['PR_AUC']:.4f}")
        print(f"   Recall: {metrics['Recall']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   F1: {metrics['F1']:.4f}")
    
    print(f"\n✅ All models saved to {output_dir}/")
    print("\nNext steps:")
    print("1. Update app.py MODEL_REGISTRY to point to these new models")
    print("2. Restart your Streamlit app")

if __name__ == '__main__':
    main()
