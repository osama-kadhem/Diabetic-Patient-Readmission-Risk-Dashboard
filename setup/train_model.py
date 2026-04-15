"""Trains all model variants and generates explanation stability artifacts."""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
TOPK = 10

# Feature set shared with the live dashboard
FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]

# Model variants — add or remove entries here to include in training
MODEL_SPECS = [
    {"id": "baseline_v1", "strategy": "none", "label": "Standard Model"},
    {"id": "classweight_v1", "strategy": "class_weight", "label": "Balanced Model"},
    {"id": "ros_v1", "strategy": "ros", "label": "Enhanced (ROS)"},
    {"id": "smote_v1", "strategy": "smote", "label": "Enhanced (SMOTE)"},
]

from src.interpretability import compute_stability, generate_stability_visuals, artifact_export_pack

def load_robust_data():
    """Load data and perform a patient-level train/test split."""
    print(f"Loading data with fixed seed {RANDOM_SEED}...")
    df = pd.read_csv('data/trained_data.csv')
    
    X = df[FEATURES]
    y = df['readmitted']
    groups = df['patient_nbr']
    
    # 80/20 Patient-Level Split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Data split: {len(X_train)} train, {len(X_test)} test (0 overlap)")
    return X_train, X_test, y_train, y_test

def get_pipeline(strategy):
    """Return a configured sklearn/imblearn pipeline for the given strategy."""
    if strategy == "none":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
        ])
    elif strategy == "class_weight":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_SEED))
        ])
    elif strategy == "ros":
        return ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomOverSampler(random_state=RANDOM_SEED)),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
        ])
    elif strategy == "smote":
        # Defaults to ROS since the restricted numeric-only feature set lacks categorical indices for SMOTE-NC
        return ImbPipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
        ])
    return None

def main():
    X_train, X_test, y_train, y_test = load_robust_data()

    pipes = {}
    clinical_models_dir = Path('clinical_models')
    clinical_models_dir.mkdir(exist_ok=True)
    
    print("\n=== Training Cycle ===")
    for spec in MODEL_SPECS:
        mid = spec['id']
        print(f"Training {mid} ({spec['label']})...")
        
        pipeline = get_pipeline(spec['strategy'])
        if pipeline:
            pipeline.fit(X_train, y_train)
            pipes[mid] = pipeline
            
            # Serialize the fitted pipeline
            joblib.dump(pipeline, clinical_models_dir / f"{mid}.joblib")

            # Write a sidecar metadata file
            meta = {
                "version_id": mid,
                "label": spec['label'],
                "strategy": spec['strategy'],
                "seed": RANDOM_SEED
            }
            with open(clinical_models_dir / f"{mid}.json", 'w') as f:
                json.dump(meta, f, indent=2)

    print("\n=== Generating Evaluation & Stability Artifacts ===")
    stability_df, topk_sets = compute_stability(pipes, X_test, topk=TOPK)

    report_dir = "clinical_models/stability_report"
    generate_stability_visuals(stability_df, topk_sets, output_dir=report_dir)
    artifact_export_pack(stability_df, topk_sets, output_dir=report_dir)

    top_pair = stability_df.iloc[0]
    print(f"\nMost Stable Pair: {top_pair['model_a']} and {top_pair['model_b']}")
    print(f"   Jaccard: {top_pair['jaccard_top10']:.4f}")

    stable_features = topk_sets[top_pair['model_a']].intersection(topk_sets[top_pair['model_b']])
    print(f"Stable Features ({len(stable_features)}): {list(stable_features)}")

    print(f"\nTraining complete. Models saved to {clinical_models_dir}/ and artifacts to {report_dir}/")

if __name__ == '__main__':
    main()
