import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path

def compute_stability(pipes, X_test, topk=10):
    """
    Computes SHAP-based global feature importance for each model and 
    calculates pairwise stability metrics (Overlap and Jaccard).
    """
    print(f"Computing stability for {len(pipes)} models using top-{topk} features...")
    
    topk_sets = {}
    stability_results = []
    
    # Getting Top-K features for models
    for model_id, pipeline in pipes.items():
        print(f"  Processing {model_id}...")
        
        # Limit sample size to keep SHAP computation fast
        X_sample = X_test.head(100) if len(X_test) > 100 else X_test
        
        # Preprocess the sample
        scaler = pipeline.named_steps['scaler']
        X_transformed = scaler.transform(X_sample)
        
        # Get feature names from test DataFrame columns
        feature_names = X_test.columns.tolist()
        
        # For LR pipelines use a LinearExplainer
        model = pipeline.named_steps['model'] if 'model' in pipeline.named_steps else pipeline.named_steps['classifier']
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        
        # Global importance is the mean absolute SHAP value
        global_importance = np.abs(shap_values).mean(0)
        
        # Create a series for ranking
        importance_series = pd.Series(global_importance, index=feature_names)
        topk_features = importance_series.sort_values(ascending=False).head(topk).index.tolist()
        
        topk_sets[model_id] = set(topk_features)
        
        # Save individual model top-K to a CSV for artifacts
        df_topk = pd.DataFrame({
            'feature': importance_series.sort_values(ascending=False).index,
            'importance': importance_series.sort_values(ascending=False).values
        }).head(topk)
        topk_sets[f"{model_id}_df"] = df_topk

    # Pairwise comparison to determine stability
    model_ids = list(pipes.keys())
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            m1, m2 = model_ids[i], model_ids[j]
            s1, s2 = topk_sets[m1], topk_sets[m2]
            
            overlap = len(s1.intersection(s2))
            union = len(s1.union(s2))
            jaccard = overlap / union if union > 0 else 0
            
            stability_results.append({
                'model_a': m1,
                'model_b': m2,
                'top10_overlap': overlap,
                'jaccard_top10': jaccard
            })
            
    stability_df = pd.DataFrame(stability_results).sort_values(by='jaccard_top10', ascending=False)
    
    return stability_df, topk_sets

def generate_stability_visuals(stability_df, topk_sets, output_dir="clinical_models/research_week_4_5"):
    """Generates heatmaps and bar charts for the stability report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap visualization generation
    model_ids = sorted(list(set(stability_df['model_a']).union(set(stability_df['model_b']))))
    n = len(model_ids)
    
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0) # Jaccard with self is 1.0
    
    for _, row in stability_df.iterrows():
        idx_a = model_ids.index(row['model_a'])
        idx_b = model_ids.index(row['model_b'])
        matrix[idx_a, idx_b] = row['jaccard_top10']
        matrix[idx_b, idx_a] = row['jaccard_top10']
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlGnBu')
    
    # Annotate
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black" if matrix[i, j] < 0.7 else "white")
            
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(model_ids, rotation=45)
    ax.set_yticklabels(model_ids)
    plt.title("Explanation Stability (Jaccard Index)")
    plt.colorbar(im)
    plt.tight_layout()
    
    img_path = Path(output_dir) / "stability_heatmap_w4_5.png"
    plt.savefig(img_path)
    plt.close()
    
    print(f"✅ Generated stability heatmap: {img_path}")

    # Driver Visuals
    for model_id in model_ids:
        df_topk = topk_sets.get(f"{model_id}_df")
        if df_topk is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot importance
            bars = ax.barh(df_topk['feature'], df_topk['importance'], color='#0284c7')
            ax.invert_yaxis() # Highest importance at top
            ax.set_xlabel('Mean |SHAP Value| (Global Importance)')
            ax.set_title(f"Global Risk Drivers: {model_id} (Test Set)")
            
            plt.tight_layout()
            driver_img_path = Path(output_dir) / f"drivers_{model_id}.png"
            plt.savefig(driver_img_path)
            plt.close()
            print(f"✅ Generated driver visual: {driver_img_path}")

def artifact_export_pack(stability_df, topk_sets, output_dir="clinical_models/research_week_4_5"):
    """Saves final CSVs and README for the week."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main stability CSV
    stab_path = Path(output_dir) / "stability_metrics_w4_5.csv"
    stability_df.to_csv(stab_path, index=False)
    
    # Save individual top-K CSVs
    for key, val in topk_sets.items():
        if key.endswith('_df'):
            fn = f"top10_{key.replace('_df', '')}.csv"
            val.to_csv(Path(output_dir) / fn, index=False)
            
    # Create README
    readme_content = f"""# Weeks 4 & 5 — Consolidated Research Pack

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Results Summary
- Most stable pair: {stability_df.iloc[0]['model_a']} vs {stability_df.iloc[0]['model_b']} (Jaccard: {stability_df.iloc[0]['jaccard_top10']:.2f})
- Least stable pair: {stability_df.iloc[-1]['model_a']} vs {stability_df.iloc[-1]['model_b']} (Jaccard: {stability_df.iloc[-1]['jaccard_top10']:.2f})

Files included:
- `stability_heatmap_w4_5.png`: Heatmap of pairwise Jaccard similarity.
- `stability_metrics_w4_5.csv`: Raw stability metrics (Weeks 4 & 5 balance check).
- `top10_<model>.csv`: Top driving features for each trained model version.
"""
    with open(Path(output_dir) / "README_W4_5.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Exported artifact pack to {output_dir}/")
