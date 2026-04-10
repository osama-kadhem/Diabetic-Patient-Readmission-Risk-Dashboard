# Clinical Readmission Risk — Final Submission

## Best model
- Version ID : lr_classweight_w7
- SHA-256    : e56f1ab7964b2d8baf8be69e8e5ec9877a0bf36ba7f1fa0a00a2d1f9d671889c
- τ_HR       : 0.604
- τ_F1       : 0.514

## Performance summary
                   PR_AUC      F1  Recall  Precision   Brier  tau_F1  tau_HR
Model                                                                       
lr_classweight_w7  0.1863  0.2444  0.5351     0.1583  0.2276   0.514   0.604
xgb_scaled_w7      0.1852  0.2447  0.4895     0.1632  0.2177   0.543   0.573

## Reproducibility
All metrics reproduced (τ=0.5) : YES
Fixed seed : RANDOM_STATE = 42
Split      : patient-grouped GroupShuffleSplit (80/20)

## Repository structure
  week7_artifacts/   — model comparison, PR curves, Jaccard heatmap
  week8_artifacts/   — calibration, ECE
  week9_artifacts/   — robustness, edge cases, sample dossiers
  week10_final/      — consolidated summary, validation, README