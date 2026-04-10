# 🏥 Clinical Readmission Risk Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![SQLite](https://img.shields.io/badge/SQLite-3.x-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org)
[![OpenFDA](https://img.shields.io/badge/OpenFDA-Live%20API-0369a1?style=for-the-badge)](https://open.fda.gov)

> A privacy-first clinical decision support system that predicts **30-day hospital readmissions** for diabetic patients. Built on calibrated Logistic Regression and the UCI Diabetes 130-US Hospitals dataset, it transforms complex patient data into actionable clinical intelligence.

---

## 🌐 Local Development

Once running locally (see [Quickstart](#-developer-quickstart) below), the dashboard is accessible at:

| Service          | Endpoint                                   |
| :--------------- | :----------------------------------------- |
| **Dashboard**    | [http://localhost:8501](http://localhost:8501) |
| **Login**        | One-click clinical authentication           |

**Default Session:** `DR. OSAMA SALEH (DEPT-CARDIO)`

---

## 🏆 Research-Led Features:

This project extends the core prediction requirement with clinically motivated features designed to demonstrate advanced data integration and professional implementation.

### ⚕️ 1. Live Polypharmacy Safety Checks (External API Integration)

The dashboard integrates with the **U.S. Government OpenFDA API** in real time. When a patient has multiple active diabetes medications (e.g. Insulin + Glipizide), the system dynamically queries the federal drug label database for registered drug-drug interactions (DDI). Detected contraindications are surfaced as colour-coded safety alerts in both the UI and exported PDFs. If the API is unreachable, a local heuristic fallback ensures patient safety is never compromised.

### 📊 2. Interactive Risk Visualisation (Altair Analytics)

The Overview tab renders an interactive **risk distribution histogram** with real-time threshold reference lines, a **donut chart** for risk-band proportions, and a **horizontal bar chart** showing per-patient feature contributions (risk drivers vs. protective factors). All charts are built with Altair for full interactivity and zoom.

### 🔐 3. Privacy & Security Hardening

- **Fernet Encryption:** All clinical notes stored in the SQLite audit log are encrypted at rest using `cryptography.fernet`. The encryption key is auto-generated on first run and stored outside the repository.
- **SHA-256 Model Integrity:** Every ML pipeline is hashed on load to detect tampering.
- **Full Audit Trail:** Every login, prediction batch, patient dossier access, and intervention is logged to SQLite for compliance.

### 📋 4. Personalised Discharge Planning

A rule-based discharge plan generator produces patient-facing letters in plain English, with personalised:
- **Diet advice** (mapped to the patient's top risk drivers)
- **Exercise recommendations** (adapted by risk band severity)
- **Medication adherence reminders** (triggered by insulin/medication changes)
- **"When to return to hospital" red-flag list**

Plans are exportable as professional **PDF letters** via `fpdf`, ready for clinical handover.

---

## 🏗️ Technical Implementation & Rationale

**Python + Streamlit:** Python's numerical ecosystem (scikit-learn, pandas, numpy) is the natural choice for clinical ML pipelines. Streamlit was chosen over Flask/Django for rapid, interactive dashboard prototyping with native support for caching (`@st.cache_resource`), session state, and reactive widgets — all critical for a clinical workflow where clinicians need instant feedback.

**Calibrated Logistic Regression:** The final model uses class-weighted Logistic Regression with Platt scaling (isotonic calibration) to ensure raw outputs represent true clinical probabilities. This was chosen over black-box models (XGBoost, neural networks) for two reasons: (1) full linear interpretability — clinicians can see exactly which features drive each prediction via coefficient×value decomposition; (2) calibrated probabilities are essential for clinical thresholds — an uncalibrated "0.6" is meaningless, but a calibrated 60% readmission probability directly informs triage decisions.

**SQLite + Fernet Encryption:** SQLite provides zero-configuration portability for a self-contained clinical prototype. All patient intervention notes are encrypted at rest with Fernet symmetric encryption, ensuring sensitive clinical data is never stored in plaintext. The encryption key (`data/.clinical_key`) is auto-generated and gitignored.

**OpenFDA Integration:** The drug-drug interaction checker queries `api.fda.gov/drug/label.json` in real time, searching for registered contraindications between the patient's active diabetes medications. This demonstrates live external API integration with graceful degradation (offline heuristic fallback).

### 📂 Directory Structure

```text
demo/
├── app.py                  # Primary clinician-facing Streamlit interface
├── clinical_models/        # Validated ML pipelines and experiment artifacts
│   ├── final_model/        # Final calibrated model, manifest, and validation data
│   └── ...                 # Iterative experiment tracking directories
├── data/                   # Patient cohort CSVs, SQLite audit DB, encryption key
├── setup/                  # Environment preparation and training scripts
│   ├── requirements.txt    # Pinned dependency index
│   └── train_model.py      # Automated baseline training pipeline
└── src/                    # Core clinical decision logic
    ├── __init__.py          # Package initialiser
    ├── data_validation.py   # Incoming CSV schema validation
    ├── db.py                # Encrypted SQLite audit and prediction logging
    ├── discharge_plan.py    # Rule-based discharge plan and PDF letter generator
    ├── interactions.py      # Live OpenFDA drug-drug interaction engine
    ├── interpretability.py  # SHAP-based global feature importance (training only)
    ├── predict.py           # Batch inference and risk ranking engine
    ├── reports.py           # Patient dossier PDF report generator
    └── risk_calculator.py   # Individual risk predictor forms and interpretation
```

---

## 📊 Core ML Concepts

### 1. Class-Weighted Logistic Regression

The UCI Diabetes dataset has a severe class imbalance (~11.2% positive readmission rate). Standard models would trivially predict "no readmission" for every patient. The system addresses this via:
- **Class weighting:** `class_weight='balanced'` inversely weights the loss function by class frequency.
- **Random Over-Sampling (ROS):** Baseline models are also trained with oversampled minority examples for comparison.

### 2. Calibrated Clinical Thresholds

Raw model outputs are post-hoc calibrated using **Platt scaling (isotonic regression)** so that a predicted probability of 0.60 truly corresponds to a 60% readmission risk. The dashboard offers two operating modes:

| Mode | Threshold | Use Case |
| :--- | :-------- | :------- |
| **Best-F1** (Default) | τ = 0.514 | Maximises overall accuracy |
| **High-Recall Screening** | τ = 0.604 | Aggressively flags at-risk patients |

### 3. Feature-Level Interpretability

For every patient, the system decomposes the prediction into individual feature contributions using `coefficient × scaled_value`. The top 10 features are displayed as a **horizontal diverging bar chart** (red = increases risk, green = protective). This replaces black-box SHAP with a fully deterministic, auditable explanation.

---

## 🏁 Developer Quickstart

### 1. Setup
```bash
git clone <repository-url>
cd demo

# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r setup/requirements.txt
```

### 2. Run Locally
```bash
# Launch the Streamlit dashboard
streamlit run app.py
```

*(Optional)* To retrain baseline models from source data:
```bash
python setup/train_model.py
```

---

## 📦 Data Schema

Uploaded CSV files must include the following columns for the final model to produce valid predictions:

| Category | Required Columns |
| :------- | :--------------- |
| **Numeric** | `time_in_hospital`, `num_medications`, `number_inpatient`, `number_emergency`, `number_outpatient` |
| **Categorical** | `age`, `A1Cresult`, `insulin`, `change`, `diabetesMed`, `metformin`, `glipizide`, `glyburide`, `max_glu_serum`, `admission_type_id`, `discharge_disposition_id` |
| **Identifiers** | `patient_nbr` or `patient_id` (recommended) |

> **Note:** The system automatically drops any target labels (`readmitted`, `readmitted_binary`, `target`) to prevent data leakage during live inference.

---

## 🔬 Academic References

1. **Strack, B. et al. (2014)** 'Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records', *BioMed Research International*. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670).
2. **UCI Machine Learning Repository (2014)** 'Diabetes 130-US Hospitals for Years 1999-2008'. Available at: [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
3. **OpenFDA (2026)** *Drug Label API*. U.S. Food and Drug Administration. Available at: [https://open.fda.gov](https://open.fda.gov).
4. **Platt, J.C. (1999)** 'Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods', *Advances in Large Margin Classifiers*.

---


*Disclaimer: This software is an academic prototype for clinical decision support research. It is not a certified medical device and must not be used to inform actual patient care or treatment decisions.*
