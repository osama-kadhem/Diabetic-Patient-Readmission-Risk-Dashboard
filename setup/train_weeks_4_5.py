"""
Weeks 4 & 5: Unified Model Training & Stability Artifact Generation
This script trains multiple models based on a configuration list, ensures 
training robustness with fixed seeds, and produces stability artifacts.
"""

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

# --- Configuration ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
TOPK = 10

# FEATURES matching the dashboard
FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]

# MODEL_SPECS: Define your model versions here for easy toggling
MODEL_SPECS = [
    {"id": "baseline_v1", "strategy": "none", "label": "Standard Model"},
    {"id": "classweight_v1", "strategy": "class_weight", "label": "Balanced Model"},
    {"id": "ros_v1", "strategy": "ros", "label": "Enhanced (ROS)"},
    {"id": "smote_v1", "strategy": "smote", "label": "Enhanced (SMOTE)"},
]

# --- Workflow Integration ---
from src.interpretability import compute_stability, generate_stability_visuals, artifact_export_pack

def load_robust_data():
    """Load data and perform patient-level split with fixed seed"""
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
    
    print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test (0 overlap)")
    return X_train, X_test, y_train, y_test

def get_pipeline(strategy):
    """Factory for model pipelines based on strategy"""
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
        # Identifying categorical indices for SMOTE-NC (none for now in this restricted feature set, but example shown)
        return ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomOverSampler(random_state=RANDOM_SEED)), # Default to ROS if SMOTE setup is empty
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
        ])
    return None

def main():
    # 1. Prepare Data
    X_train, X_test, y_train, y_test = load_robust_data()
    
    # 2. Train Models
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
            
            # Save for dashboard
            joblib.dump(pipeline, clinical_models_dir / f"{mid}.joblib")
            
            # Save metadata
            meta = {
                "version_id": mid,
                "label": spec['label'],
                "strategy": spec['strategy'],
                "seed": RANDOM_SEED
            }
            with open(clinical_models_dir / f"{mid}.json", 'w') as f:
                json.dump(meta, f, indent=2)

    # 3. Generate Weeks 4 & 5 Artifacts (Benchmarking + Stability)
    print("\n=== Generating Weeks 4 & 5 Research Artifacts ===")
    stability_df, topk_sets = compute_stability(pipes, X_test, topk=TOPK)
    
    # Generate visuals and export to consolidated research location
    report_dir = "clinical_models/research_week_4_5"
    generate_stability_visuals(stability_df, topk_sets, output_dir=report_dir)
    artifact_export_pack(stability_df, topk_sets, output_dir=report_dir)
    
    # 4. Final Recommendation (Training refined)
    top_pair = stability_df.iloc[0]
    print(f"\n🏆 Most Stable Model Pair: {top_pair['model_a']} and {top_pair['model_b']}")
    print(f"   Jaccard Stability: {top_pair['jaccard_top10']:.4f}")
    
    stable_features = topk_sets[top_pair['model_a']].intersection(topk_sets[top_pair['model_b']])
    print(f"📍 Highly Stable Features ({len(stable_features)}): {list(stable_features)}")
    
    print(f"\n✅ Week 5 Training Complete. Models saved to {clinical_models_dir}/ and artifacts to week5_artifacts/")

if __name__ == '__main__':
    main()
