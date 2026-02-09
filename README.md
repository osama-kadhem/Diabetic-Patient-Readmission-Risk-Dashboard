# Patient Readmission Dashboard

A professional data tool for predicting 30-day hospital readmission risk specifically for diabetic patients. This application provides clinicians with a list of high-risk patients and detailed analysis of risk factors to assist in discharge planning.

## 🚀 Quick Start

1. **Install Requirements**
   ```bash
   pip install -r setup/requirements.txt
   ```

2. **Initialize Model & Data**
   ```bash
   python setup/train_model.py
   ```
   *This trains the prediction model and generates `data/diabetes_test_data.csv`.*

3. **Launch Dashboard**
   ```bash
   streamlit run app.py
   ```

## 🛠 Features

- **Predictive Scoring:** Uses a logistic regression model to calculate readmission probability.
- **Risk Stratification:** Automatically groups patients into High, Medium, and Low risk bands.
- **Dossier View:** Deep-dive into individual patient records and key clinical metrics.
- **Intervention Logging:** Securely record clinician actions and notes for each patient.
- **Data Privacy:** Local processing with persistent audit logs of data access.

## 📂 Project Structure

- `app.py`: Main dashboard application interface.
- `setup/`: Setup scripts and configuration.
    - `train_model.py`: Training script and synthetic data generator.
    - `requirements.txt`: Python package dependencies.
- `data/`: Contains database and sample CSV files.

- `src/`: 
    - `predict.py`: Core prediction and ranking logic.
    - `db.py`: Secure database and audit logging.
- `models/`: Storage for the trained model pipeline.

## ⚖️ Disclaimer
This dashboard is a demonstration tool for data analysis and clinical decision support. It should be used as a supplement to professional medical judgment.
