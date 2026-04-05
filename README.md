# Clinical Readmission Risk Dashboard

**Anticipate. Intervene. Care.**

A remarkably intelligent, privacy-first clinical decision support system designed specifically to predict 30-day hospital readmissions for diabetic patients. Built on a foundation of rigorous machine learning, it places actionable insights instantly into the hands of clinicians, transforming complex data into clear, life-saving foresight.

---

## Uncompromising Intelligence
Powered by a calibrated, class-weighted Logistic Regression engine, the dashboard evaluates complex patient demographics, medication histories, and admission data in milliseconds. It cuts through severe data imbalance, ensuring the most vulnerable patients are never overlooked.

- **Intelligent Risk Stratification:** Automatically groups patient encounters into High, Moderate, and Low risk bands based on meticulously optimized clinical thresholds.
- **Deep Interpretability:** See the "why" behind every prediction. The Patient Dossier intuitively surfaces the top ten features driving a patient's risk score, replacing algorithmic black-boxes with complete clinical transparency.
- **Live Polypharmacy Safety Checks:** Seamlessly integrated with the U.S. Government's OpenFDA clinical API. The system dynamically queries real-time federal databases to detect severe drug-drug interactions (DDI) directly from the patient's active medication list, appending critical safety alerts to the clinician's dossier.
- **Dynamic What-If Analysis:** Interactively model discharge scenarios. Adjust medications and prior admission histories to instantly witness the projected impact on patient risk in real time.
- **Tiered Discharge Planning:** Automatically generate and export high-fidelity, secure PDF discharge dossiers outlining targeted intervention strategies for at-risk patients.
- **Frictionless Workflow:** Seamlessly upload standard HL7/CSV encounter extracts, define your daily intervention capacity, and let the system intelligently prioritize your team's follow-up queue.

## Privacy by Design
In healthcare, trust is paramount. The system is engineered to operate efficiently with uncompromising focus on data security and compliance.

- **Local Autonomy:** Zero reliance on an external cloud. All sensitive patient data is processed strictly locally within your institution's secure network environment.
- **Cryptographic Assurance:** Every machine learning model is subjected to an SHA-256 integrity verification upon load, ensuring the clinical logic has never been compromised or tampered with.
- **Auditable Accountability:** A robust, onboard SQLite architecture silently logs every prediction batch, session authentication, and patient dossier interaction, enabling seamless compliance reporting and post-hoc clinical reviews.

## Built on Scientific Rigor
The application doesn't just predict; it proves its accuracy through empirical validation.

- **Imbalance Resilience:** Developed specifically to handle severe class imbalances (~11.2% positive rate) using advanced class-weighting and random oversampling techniques across an 80/20 patient-grouped data split.
- **Calibrated Precision:** Employs Platt scaling (Isotonic regression) to ensure raw model outputs represent true clinical probabilities.
- **Dual Operating Modes:** Swap seamlessly between the **Best-F1** point (maximizing general accuracy) and the **High-Recall Screening** point (aggressively flagging at-risk patients to leave no stone unturned).

---

## Getting Started

Everything you need to deploy the platform locally, streamlined into a few elegant steps.

### 1. Prerequisites
Ensure you have the following installed on your machine:
- Python 3.10 or later
- pip (Python package installer)

### 2. Setup Your Environment
Clone the repository and install the precisely defined dependencies:
```bash
# Navigate to the project directory
cd final-project-demo

# Install the required packages
pip install -r setup/requirements.txt
```

### 3. Launch the Application
Start the Streamlit environment. The dashboard will instantly become available in your default web browser:
```bash
streamlit run app.py
```
*(Optional)* To regenerate synthetic training data and establish a baseline model locally, run `python setup/train_model.py` prior to launching the dashboard.

---

## 🏥 The Clinical Workflow

Beautiful software should be fundamentally simple to use. Here is how your clinical team will interact with the system on a daily basis:

1. **Secure Authentication:** Log into the immutable session state. Your interactions and decisions are securely bound to your session.
2. **Ingest Patient Data:** Upload an exported CSV file of recent diabetic encounters from your hospital's Electronic Health Record (EHR) database.
3. **Configure Constraints:** Set the team's intervention capacity for the day (e.g., 20 patients). The system adjusts its prioritization algorithms accordingly.
4. **Run Analysis:** The engine processes the batch, applying SHA-256 validated models to score and rank patients.
5. **Review the Queue:** Focus immediately on the High-Risk band in the Prioritization Queue.
6. **Export Actionable Plans:** Dive into the Patient Dossier to analyze feature impacts, run What-If simulations, and export a generated PDF action plan directly to the discharging physician.

---

## Data Schema Expectations

To ensure flawless analysis, uploaded CSV files must include the following structural features:
- **Identifiers:** `patient_id` or `patient_nbr`
- **Clinical Numerics:** `time_in_hospital`, `num_medications`, `number_inpatient`, `number_outpatient`, `number_emergency`.
- **Demographics & Categories:** `age`, `A1Cresult`, `admission_type_id`, `discharge_disposition_id`.
- **Medication Indicators:** `insulin`, `change`, `diabetesMed`, `metformin`, `glipizide`, `glyburide`.

*Note: The system intelligently drops any target labels (`readmitted` or `target`) to prevent data leakage during live inference.*

---

## Project Architecture

Elegant design extends deeply into the codebase. Modules are strictly delineated to separate logic, data, and presentation.

```text
final-project-demo/
├── app.py                  # Primary clinician-facing Streamlit interface
├── clinical_models/        # Cryptographically hashed, calibrated ML assets
│   ├── week10_final/       # Final deployment models and validation manifests
│   └── ...                 # Iterative experiment tracking directories
├── data/                   # Patient cohort CSVs and generated datasets
├── setup/                  # Environment preparation and synthetic data scripts
│   ├── requirements.txt    # Project dependency index
│   └── train_model.py      # Automated baseline training pipeline
└── src/                    # Core clinical decision logic and utilities
    ├── data_validation.py  # Incoming data schema assurance & verification
    ├── db.py               # Secure SQLite audit and prediction logging backend
    ├── discharge_plan.py   # Secure, exportable PDF intervention generation
    ├── interactions.py     # Live OpenFDA API DDI engine & safety heuristics
    ├── predict.py          # Core inference mapping and risk ranking engine
    ├── reports.py          # Supplementary patient dossier formatting integrations
    └── risk_calculator.py  # What-If simulation forms & engine constraint mapping
```

---

*Disclaimer: This software is designed as a sophisticated demonstrative tool for data analysis and clinical decision support. It is engineered to augment, not replace, professional medical judgment.*
