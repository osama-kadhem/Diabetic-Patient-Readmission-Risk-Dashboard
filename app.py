import streamlit as st

st.set_page_config(  # must be the first Streamlit call
    page_title="Clinical Dashboard - Readmission Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import io
import altair as alt
import hashlib

from src.data_validation import validate_csv
from src.predict import predict_risk, rank_patients
from src.reports import generate_patient_pdf
import src.db as db
from src.risk_calculator import (
    load_manifest, load_model,
    render_patient_form, compute_risk_band, interpret_risk,
    MANIFEST_PATH, _patch_lr_compat
)
from src.discharge_plan import generate_discharge_plan, generate_patient_discharge_pdf
from src.interactions import check_drug_interactions

@st.cache_data
def get_local_explanation(pipeline_hash, _pipeline, patient_row):
    """Compute per-feature contributions for a patient using the active pipeline."""
    try:
        try:
            expected_cols = list(_pipeline.feature_names_in_)
        except AttributeError:
            expected_cols = [
                'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses'
            ]
            
        missing = [c for c in expected_cols if c not in patient_row.index]
        if missing:
            return pd.DataFrame({'Error': [f"Missing feature columns: {missing[:3]}..."]})
            
        # Resolve the preprocessor step (varies by pipeline version).
        preprocessor = (
            _pipeline.named_steps.get('preprocess') or
            _pipeline.named_steps.get('preprocessor') or
            _pipeline.named_steps.get('scaler')
        )
        if preprocessor is None:
            return pd.DataFrame({'Error': ["Pipeline does not expose a preprocessing step."]})

        model = _pipeline.named_steps.get('classifier') or _pipeline.named_steps.get('model')
        if model is None:
            return pd.DataFrame({'Error': ["No 'model' or 'classifier' step found in pipeline."]})

        X = patient_row[expected_cols].to_frame().T
        X_scaled = preprocessor.transform(X)
        if hasattr(X_scaled, 'toarray'):
            X_scaled = X_scaled.toarray()

        if hasattr(model, 'coef_'):
            weights = model.coef_[0]
            contributions = X_scaled[0] * weights
        elif hasattr(model, 'feature_importances_'):
            weights = model.feature_importances_
            contributions = X_scaled[0] * weights
        else:
            return pd.DataFrame({'Error': ["Model object has neither coef_ nor feature_importances_."]})

        try:
            out_names = preprocessor.get_feature_names_out()
        except Exception:
            out_names = expected_cols

        if len(contributions) != len(out_names):
            out_names = [f"Feature {i}" for i in range(len(contributions))]

        name_map = {
            'Time In Hospital':   'Days in Hospital',
            'Num Lab Procedures': 'Lab Tests Performed',
            'Num Procedures':     'Clinical Procedures',
            'Num Medications':    'Active Medications',
            'Number Outpatient':  'Prior Outpatient Visits',
            'Number Emergency':   'Prior Emergency Visits',
            'Number Inpatient':   'Prior Inpatient Admissions',
            'Number Diagnoses':   'Comorbidity Count',
        }

        # Format column names for display
        feature_labels = [
            name_map.get(
                str(c).split('__')[-1].replace('_', ' ').title(), 
                str(c).split('__')[-1].replace('_', ' ').title()
            ) for c in out_names
        ]

        df_exp = pd.DataFrame({
            'Feature': feature_labels,
            'Contribution': contributions,
        }).sort_values(by='Contribution', key=abs, ascending=False)
        
        return df_exp.head(10)

    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

db.init_db()



# Custom CSS - forces light theme and sets the font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap');

    /* Main app background and text */
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"],
    .main .block-container {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
    }

    /* Text colour for markdown elements */
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #1e293b !important;
        font-family: 'Public Sans', sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
        font-family: 'Public Sans', sans-serif !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #0369a1 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem !important;
    }

    /* Cards */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: #ffffff !important;
        border-radius: 4px !important;
        border-left: 4px solid #0284c7 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Public Sans', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #0369a1 !important;
    }

    /* Input Fields */
    .stTextInput input, .stNumberInput input,
    [data-testid="textInput"] input {
        border-radius: 4px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-bottom: 2px solid #e2e8f0;
        padding: 0 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748b !important;
        padding: 10px 20px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 2px;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
    }

    .risk-high     { background-color: #dc2626; color: #ffffff !important; }
    .risk-moderate { background-color: #d97706; color: #ffffff !important; }
    .risk-low      { background-color: #059669; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Model Registry
MODEL_REGISTRY = {
  "lr_classweight_w7_final": "clinical_models/final_model/lr_classweight_w7_final.joblib",
  "lr_classweight_w7":       "clinical_models/research_models/lr_classweight_w7.pkl",
  "lr_ros_w6":               "clinical_models/candidate_models/lr_ros_w6.joblib"
}

MODEL_LABELS = {
  "lr_classweight_w7_final": "Calibrated LR (FINAL - Default)",
  "lr_classweight_w7":       "LR Class-Weight (Uncalibrated)",
  "lr_ros_w6":               "LR Random Over-Sampling (Baseline)"
}

# Cache-bust version
_CACHE_VERSION = 5

# Set up session state keys so they always exist
_session_defaults = {
    'authenticated': False,
    'user': None,
    'uploaded_data': None,
    'predictions': None,
    'selected_patient': None,
    'pipeline': None,
    'model_version': "lr_classweight_w7_final",
    'risk_reset_key': 0,
    'risk_result': None,
    'discharge_plan_text': None,
}
for _k, _v in _session_defaults.items():
    st.session_state.setdefault(_k, _v)

def compute_model_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a model file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_model_integrity(file_path: str) -> str:
    """
    Verify model integrity via SHA-256. On first load the hash is stored as a
    sidecar .sha256 file. On subsequent loads the recomputed hash is compared
    against the stored value; a mismatch raises RuntimeError before any
    prediction is served.
    """
    model_path = Path(file_path)
    hash_path  = model_path.with_suffix(".sha256")
    current    = compute_model_hash(file_path)

    # Note: st.session_state is accessed carefully because this function might
    # run before proper login in some contexts, so user is "SYSTEM" if unknown.
    # Safely get current user
    try:
        user = st.session_state.get('user_name', 'SYSTEM')
    except Exception:
        user = "SYSTEM"

    if hash_path.exists():
        stored = hash_path.read_text().strip()
        if stored != current:
            db.log_security_event(user, "MODEL_INTEGRITY_CHECK", str(model_path.name), "FAILED_HASH_MISMATCH")
            raise RuntimeError(
                f"SHA-256 integrity check FAILED for {model_path.name}. "
                "The model file may have been tampered with or corrupted. "
                f"Stored: {stored[:12]}…  Current: {current[:12]}…"
            )
        else:
            db.log_security_event(user, "MODEL_INTEGRITY_CHECK", str(model_path.name), "PASSED")
    else:
        # First load — store hash as trust-on-first-use baseline
        hash_path.write_text(current)
        db.log_security_event(user, "MODEL_INTEGRITY_BASELINE", str(model_path.name), "HASH_CREATED")

    return current

def login_screen():
    st.markdown("""
<div style='text-align: center; padding: 50px 0;'>
<h1 style='color: #0369a1;'>LOGIN</h1>
<p style='color: #64748b;'>Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)
    
    with st.container(border=True):
        if st.button("LOGIN", type="primary", use_container_width=True):
            # Mark the session as authenticated and log the event
            st.session_state.authenticated = True
            st.session_state.user = "DR. OSAMA SALEH"
            db.log_audit(st.session_state.user, "LOGIN")
            st.rerun()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# Top Navigation: Clinical Information Header
st.markdown(f"""
<div style="background-color: #ffffff; padding: 10px 20px; border-bottom: 2px solid #0284c7; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
<div style="flex: 1;">
<span style="font-weight: 800; color: #0f172a; font-size: 1.2rem;">DASHBOARD</span>
<span style="color: #64748b; font-size: 0.8rem; margin-left: 10px;"></span>
</div>
<div style="text-align: right; color: #64748b; font-size: 0.8rem; flex: 1;">
<div><b>AUTHENTICATED:</b> {st.session_state.user} (DEPT-CARDIO)</div>
<div>SESSION: {pd.Timestamp.now().strftime('%Y-%m-%d')} | LOCAL TIME: {pd.Timestamp.now().strftime('%H:%M')}</div>
</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(version, _cache_version=_CACHE_VERSION):
    """Load the active model pipeline."""
    try:
        model_path = Path(MODEL_REGISTRY[version])
        
        if not model_path.exists():
            st.error(f"ERR-404: Model file not found at {model_path}.")
            return None

        try:
            model_hash = check_model_integrity(str(model_path))
            st.caption(f"SHA-256 verified: `{model_hash[:16]}…`")
        except RuntimeError as integrity_err:
            st.error(f"INTEGRITY FAILURE: {integrity_err}")
            return None

        with open(model_path, 'rb') as f:
            pipeline = joblib.load(f)
        # Patch LogisticRegression compatibility across sklearn versions.
        _patch_lr_compat(pipeline)
        return pipeline
    except Exception as e:
        st.error(f"ERR-500: System error loading {version}: {str(e)}")
        return None

@st.cache_data
def load_and_validate_data(file_obj):
    # Cache the raw CSV so re-uploads don't re-read from disk
    df = pd.read_csv(file_obj)
    return df

with st.sidebar:
    st.markdown("### ACCOUNT & PRIVACY")
    if st.button("SECURE LOGOUT"):
        db.log_audit(st.session_state.user, "LOGOUT")
        st.session_state.authenticated = False
        st.rerun()
        
    st.markdown("---")
    st.markdown("### SYSTEM CONFIGURATION")
    
    st.markdown("#### MODEL VERSION")
    
    st.markdown("""
        <style>
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            user-select: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    selected_version = st.selectbox(
        "Active Clinical Logic",
        options=list(MODEL_REGISTRY.keys()),
        format_func=lambda x: MODEL_LABELS.get(x, x),
        index=0 if st.session_state.model_version not in MODEL_REGISTRY else list(MODEL_REGISTRY.keys()).index(st.session_state.model_version),
        help="Select the validated clinical model for risk assessment. Each version uses different training strategies optimized for specific use cases."
    )
    
    if selected_version != st.session_state.model_version:
        st.session_state.model_version = selected_version
        st.session_state.pipeline = None
        st.success(f"Model updated: {MODEL_LABELS[selected_version]}")

    st.markdown("#### DATA IMPORT")

    uploaded_file = st.file_uploader(
        "Upload Patient Encounter CSV",
        type=['csv'],
        help="Requires standard HL7 extract format (CSV)"
    )
    
    if uploaded_file is not None:
        try:
            df_full    = load_and_validate_data(uploaded_file)
            total_rows = len(df_full)
            if total_rows > 10000:
                st.warning(f"Large File: {total_rows:,} records. Sampling data...")

                use_smart_sampling = st.checkbox("Preserve Patient History (Recommended)", value=True, help="Ensures all encounters for a sampled patient are included.")
                
                if use_smart_sampling and 'patient_nbr' in df_full.columns:
                    unique_patients = df_full['patient_nbr'].unique()
                    sample_size = st.number_input("Target Patient Cohort Size", 500, len(unique_patients), 2000, 500)
                    
                    sampled_ids = np.random.choice(unique_patients, size=sample_size, replace=False)
                    df = df_full[df_full['patient_nbr'].isin(sampled_ids)].copy()
                    st.caption(f"Ingested {len(df):,} records for {sample_size:,} unique patients.")
                else:
                    sample_size = st.number_input("Generic Record Sample Size", 1000, total_rows, 10000, 1000)
                    df = df_full.sample(n=sample_size, random_state=42) if sample_size < total_rows else df_full
            else:
                df = df_full
            
            validation_result = validate_csv(df)
            
            if validation_result['is_valid']:
                if 'patient_nbr' in df.columns:
                    df['patient_id'] = df['patient_nbr']
                    
                st.session_state.uploaded_data = df
                st.success(f"Loaded {len(df):,} records.")

            else:
                st.error("VALIDATION ERROR: Internal data integrity check failed.")
                st.session_state.uploaded_data = None
        except Exception as e:
            st.error(f"IO ERROR: {str(e)}")
            st.session_state.uploaded_data = None
    
    st.markdown("---")
    
    st.markdown("#### OPERATING MODE")
    op_mode = st.radio(
        "Select clinical threshold strategy",
        ["Best-F1 (Default)", "High-Recall (Screening)"],
        index=0,
        help="Best-F1 maximizes overall accuracy. High-Recall flags more patients for review."
    )
    if "Screening" in op_mode:
        # High-Recall: lower thresholds intentionally to maximise sensitivity —
        # more patients are flagged as Moderate/High to minimise missed cases.
        st.session_state.op_mode_name = "screening"
        st.session_state.tau_high = 0.50
        st.session_state.tau_mid  = 0.35
    else:
        # Best-F1: validated operating point from calibration curve analysis.
        st.session_state.op_mode_name = "best_f1"
        st.session_state.tau_high = 0.604
        st.session_state.tau_mid  = 0.514

    st.markdown("#### CAPACITY")

    if st.session_state.uploaded_data is not None:
        max_patients = len(st.session_state.uploaded_data)
        top_k = st.number_input(
            " Daily Intervention Capacity",
            min_value=1, 
            max_value=max_patients,
            value=min(20, max_patients),
            help="Max patients team can contact today"
        )
    else:
        top_k = st.number_input("Daily Intervention Capacity", value=20, disabled=True)
    
    st.markdown("---")
    if st.button("RUN ANALYSIS", type="primary", use_container_width=True):

        if st.session_state.uploaded_data is not None:
            if st.session_state.pipeline is None:
                st.session_state.pipeline = load_pipeline(st.session_state.model_version)
            
            if st.session_state.pipeline is not None:
                with st.spinner("Analyzing..."):
                    # Drop target columns to prevent leakage
                    _TARGET_COLS = ['readmitted', 'readmitted_binary', 'label', 'target']
                    _leaky = [c for c in _TARGET_COLS
                              if c in st.session_state.uploaded_data.columns]
                    data_for_pred = (
                        st.session_state.uploaded_data.drop(columns=_leaky)
                        if _leaky
                        else st.session_state.uploaded_data
                    )

                    try:
                        predictions = predict_risk(
                            data_for_pred,
                            st.session_state.pipeline,
                            threshold_high=st.session_state.tau_high,
                            threshold_medium=st.session_state.tau_mid
                        )
                        st.session_state.predictions = rank_patients(predictions)
                    except Exception as pred_err:
                        st.error(f"⚠️ PREDICTION FAILED: {str(pred_err)}")
                        st.stop()
                    
                    try:
                        threshold  = 0.5
                        batch_data = []
                        for _, row in st.session_state.predictions.iterrows():
                            prob = row['risk_probability']
                            label = 1 if prob >= threshold else 0
                            pid = row.get('encounter_id', row.get('patient_id', 'UNKNOWN'))
                            batch_data.append((pid, st.session_state.model_version, prob, label, threshold))
                        
                        db.log_predictions_batch(batch_data)
                        st.success(f"Analysis Complete & Logged ({len(st.session_state.predictions)} records in batch)")
                    except Exception as e:
                        st.warning(f"Analysis complete but logging failed: {e}")
        else:
            st.error("Please import data first.")

    st.caption("🟢 **System Online**")
    st.caption(f"v1.0 | {pd.Timestamp.now().strftime('%Y-%m-%d')}")

# Four main tabs - first three need data before they're useful
tab1, tab2, tab3, tab4 = st.tabs(["OVERVIEW", "PRIORITIZATION QUEUE", "PATIENT DOSSIER", "RISK PREDICTOR"])

if st.session_state.uploaded_data is None:
    with tab1:
        # Welcome screen
        st.info("Ready: Please import data to start.")
        st.markdown("### HOW TO USE")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. Upload Patient Data**  
            Upload a CSV file containing patient encounter information with all required model features.

            **2. Calculate Risk Scores**  
            The system analyzes each patient and assigns a readmission risk probability (0-1).

            **3. Set Follow-up Capacity**  
            Tell us how many patients your team can contact, and we'll prioritize accordingly.
            """)
        with col2:
            st.markdown("""
            **4. Review Rankings**  
            See patients sorted by risk probability based on the integrated inference pipeline.

            **5. View Patient Details**  
            Click on any patient to see key clinical metrics for the current encounter.

            **6. System Integrity**  
            All models are verified via SHA-256 hashes on load to ensure medical data safety.
            """)
        st.markdown("### 📊 Sample Data Format")
        st.markdown("Your CSV should include patient identifiers and clinical features. Example:")
        sample_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'age': [65, 72, 58],
            'num_medications': [8, 12, 5],
            'num_diagnoses': [3, 5, 2],
            'time_in_hospital': [4, 7, 2],
            'num_procedures': [1, 3, 0],
        })
        st.dataframe(sample_data, use_container_width=True)
    with tab2:
        st.info("Ready: Please import data to start.")
    with tab3:
        st.info("Ready: Please import data to start.")

if st.session_state.uploaded_data is not None:
    with tab1:
        # Overview tab
        st.markdown("### OVERVIEW")

        if st.session_state.predictions is not None:
            df_pred = st.session_state.predictions

            # Warn if the user uploaded the full training dataset (>10k rows),
            # which causes inflated probabilities due to train-set memorisation.
            if len(df_pred) > 10_000:
                st.warning(
                    f"⚠️ **Training Data Detected ({len(df_pred):,} rows):** You appear to have uploaded the "
                    "full training dataset. The model has already seen these patients during training, so "
                    "predicted probabilities will be inflated (e.g. 98%). "
                    "**Upload a fresh prospective cohort (held-out test set or real patients) for valid predictions.**"
                )
            # Summary Metrics Container
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Patients", len(df_pred), help="Total number of patients in the uploaded file")
                
                with col2:
                    avg_risk = df_pred['risk_probability'].mean()
                    st.metric("Average Risk", f"{avg_risk:.3%}", help="Mean risk score across all patients")
                
                with col3:
                    high_risk_count = len(df_pred[df_pred['risk_band'] == 'High'])
                    if high_risk_count > 0:
                        st.metric("High Risk Patients", high_risk_count, delta="Action Required", delta_color="inverse")
                    else:
                        st.metric("High Risk Patients", high_risk_count, help="No patients currently in the high-risk band")
                
                with col4:
                    st.metric("Capacity Target", top_k, help="Number of patients targeted for follow-up")
            
            # Charts Row
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### Risk Distribution")
                    
                    # Threshold reference lines
                    thresh_df = pd.DataFrame([
                        {'threshold': st.session_state.tau_mid, 'label': 'Moderate threshold'},
                        {'threshold': st.session_state.tau_high, 'label': 'High threshold'},
                    ])
                    thresh_lines = alt.Chart(thresh_df).mark_rule(
                        strokeDash=[5, 3], strokeWidth=1.5
                    ).encode(
                        x=alt.X('threshold:Q'),
                        color=alt.Color('label:N', scale=alt.Scale(
                            domain=['Moderate threshold', 'High threshold'],
                            range=['#d97706', '#dc2626']
                        ), legend=alt.Legend(title='Thresholds'))
                    )

                    # Histogram coloured by risk band
                    risk_bars = alt.Chart(df_pred).mark_bar(opacity=0.85).encode(
                        x=alt.X(
                            'risk_probability:Q',
                            bin=alt.Bin(maxbins=30),
                            title='Risk Probability',
                            scale=alt.Scale(domain=[0, 1])   # always show full 0→1 axis
                        ),
                        y=alt.Y('count()', title='Number of Patients'),
                        color=alt.Color(
                            'risk_band:N',
                            scale=alt.Scale(
                                domain=['High', 'Moderate', 'Low'],
                                range=['#dc2626', '#d97706', '#059669']
                            ),
                            legend=alt.Legend(title='Risk Band')
                        ),
                        tooltip=[
                            alt.Tooltip('count()', title='Patients'),
                            alt.Tooltip('risk_probability:Q', bin=True, title='Probability Range', format='.2f'),
                            'risk_band:N'
                        ]
                    )

                    risk_chart = (risk_bars + thresh_lines).resolve_scale(
                        color='independent'
                    ).properties(height=300)
                    st.altair_chart(risk_chart, use_container_width=True)
                    
                    def get_band_color(prob):
                        if prob >= st.session_state.tau_high: return "#dc2626"
                        if prob >= st.session_state.tau_mid: return "#d97706"
                        return "#059669"

                    avg_p = df_pred['risk_probability'].mean()
                    max_p = df_pred['risk_probability'].max()
                    
                    for label, val in [("AVERAGE", avg_p), ("PEAK", max_p)]:
                        color = get_band_color(val)
                        st.markdown(
                            f'<div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">'
                            f'<span class="risk-badge" style="background-color:{color}; color:white; border:none;">{label}</span>'
                            f'<span style="font-weight:600;">{val:.3%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            with col2:
                with st.container(border=True):
                    st.markdown("#### Risk Bands")
                    band_counts = df_pred['risk_band'].value_counts()
                    
                    # Donut chart for bands
                    band_data = pd.DataFrame({
                        'Band': band_counts.index,
                        'Count': band_counts.values
                    })
                    
                    donut = alt.Chart(band_data).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Band", type="nominal", scale=alt.Scale(
                            domain=['High', 'Moderate', 'Low'],
                            range=['#dc2626', '#d97706', '#059669']
                        )),
                        tooltip=['Band', 'Count']
                    ).properties(height=300)
                    
                    st.altair_chart(donut, use_container_width=True)
                    
                    for band in ['High', 'Moderate', 'Low']:
                        if band in band_counts.index:
                            count = band_counts[band]
                            pct = count / len(df_pred) * 100
                            color_class = f"risk-{band.lower()}"
                            st.markdown(
                                f'<div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">'
                                f'<span class="risk-badge {color_class}">{band.upper()}</span>'
                                f'<span style="font-weight:600;">{count} ({pct:.3f}%)</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            
            # Data preview
            st.markdown("### DATA PREVIEW")

            display_cols = ['patient_id', 'risk_probability', 'risk_band', 'follow_up_priority']
            preview_df = df_pred[display_cols].head(10)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("Ready: Run analysis to start.")
            st.subheader("Data Preview")

            st.dataframe(st.session_state.uploaded_data.head(10), use_container_width=True)
    
    with tab2:
        st.markdown("### PATIENT LIST")

        if st.session_state.predictions is not None:
            df_pred = st.session_state.predictions

            with st.container(border=True):
                
                # Filters and search
                st.markdown("#### SEARCH")

                col1, col2, col3 = st.columns([2, 5, 2])
                
                with col1:
                    search_patient = st.text_input("SEARCH ID", placeholder="Search...")

                with col2:
                    risk_filter = st.multiselect(
                        "Risk Bands",
                        options=['High', 'Medium', 'Low'],
                        default=['High', 'Medium', 'Low']
                    )
                
                with col3:
                    show_top_k_only = st.checkbox("Show Top K Only", value=True, help=f"Limit to top {top_k} patients")
                
            filtered_df = df_pred.copy()
            if search_patient:
                filtered_df = filtered_df[
                    filtered_df['patient_id'].astype(str).str.contains(search_patient, case=False)
                ]
            if risk_filter:
                filtered_df = filtered_df[filtered_df['risk_band'].isin(risk_filter)]
            if show_top_k_only:
                filtered_df = filtered_df.head(top_k)
            
            with st.container(border=True):
                
                col_header1, col_header2 = st.columns([1, 1])
                with col_header1:
                    st.markdown(f"#### Patient List ({len(filtered_df)} matches)")
                
                st.dataframe(
                    filtered_df[['patient_id', 'risk_probability', 'risk_band', 'follow_up_priority']],
                    column_config={
                        "patient_id": "Patient ID",
                        "risk_probability": st.column_config.ProgressColumn(
                            "Readmission Probability",
                            help="Predicted probability of 30-day readmission",
                            format="%.1f%%",
                            min_value=0,
                            max_value=1,
                        ),
                        "risk_band": "Risk Level",
                        "follow_up_priority": st.column_config.NumberColumn(
                            "Priority Rank",
                            help="Recommended follow-up order"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

            with st.container():
                st.markdown("#### View Patient Details")
                patient_ids = filtered_df['patient_id'].tolist()
                col_sel1, col_sel2 = st.columns([3, 1])
                
                with col_sel1:
                    selected = st.selectbox(
                        "Select a patient to view full analysis:",
                        options=patient_ids,
                        index=0 if patient_ids else None
                    )
                
                with col_sel2:
                    if selected:
                        st.session_state.selected_patient = selected
                        db.log_audit(st.session_state.user, "PATIENT_DOSSIER_ACCESSED", str(selected))
                        st.success("Loaded.")

        else:
            st.info("Ready: Run analysis to start.")

    with tab3:
        st.markdown("### PATIENT DETAILS")

        if st.session_state.predictions is not None and st.session_state.selected_patient is not None:
            patient_id = st.session_state.selected_patient
            df_pred = st.session_state.predictions

            patient_row = df_pred[df_pred['patient_id'] == patient_id].iloc[0]
            patient_history = db.get_patient_history(patient_id)

            exp_df = pd.DataFrame()
            active_pipe = st.session_state.get('pipeline')
            if active_pipe is None:
                active_pipe = load_pipeline(st.session_state.model_version)
            
            if active_pipe:
                pipeline_hash = joblib.hash(active_pipe)
                exp_df = get_local_explanation(pipeline_hash, active_pipe, patient_row)

            # Calculate API Polypharmacy alerts early to pass into PDF generators 
            patient_meds = patient_row.to_dict()
            interaction_alerts = check_drug_interactions(patient_meds)

            # Hero Section
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([1.5, 1, 2, 1.5])
                
                with col1:
                    st.markdown(f"**Patient ID**")
                    st.markdown(f"## {patient_id}")
                
                with col2:
                    risk_prob = patient_row['risk_probability']
                    risk_band = patient_row['risk_band']
                    color_class = f"risk-{risk_band.lower()}"
                    
                    st.markdown(f"**Risk Band**")
                    st.markdown(
                        f'<div style="margin-top:5px;"><span class="risk-badge {color_class}">{risk_band} Risk</span></div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    priority = patient_row['follow_up_priority']
                    st.markdown(f"**30-Day Readmission Probability**")
                    st.markdown(f"<h2 style='color: #0ea5e9;'>{risk_prob:.1%}</h2>", unsafe_allow_html=True)
                    st.caption(f"Follow-up Priority Rank: #{priority}")
                
                with col4:
                    st.markdown("**Export Dossier**")
                    pdf_buffer = generate_patient_pdf(
                        patient_row, 
                        explanation_df=exp_df, 
                        history_df=patient_history,
                        user_name=st.session_state.user,
                        interaction_alerts=interaction_alerts
                    )
                    st.download_button(
                        label="📄 EXPORT PDF",
                        data=pdf_buffer,
                        file_name=f"patient_report_{patient_id}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )

                # Clinical Indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        st.markdown("#### CLINICAL SNAPSHOT")
                        st.write("Displaying key metrics for the current encounter.")
                        
                        metrics_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures', 'num_procedures']
                        metrics_cols = [c for c in metrics_cols if c in patient_row]
                        
                        if metrics_cols:
                            for col in metrics_cols:
                                st.write(f"**{col.replace('_', ' ').title()}:** {patient_row[col]}")
                        else:
                            st.info("No numeric clinical metrics available for this record.")
                
                with col2:
                    with st.container(border=True):
                        st.markdown("#### STATUS")
                        st.write(f"Admission Status: **Stable**")
                        st.write(f"Provider: **{st.session_state.user}**")
                        st.write(f"Last Interaction: **{pd.Timestamp.now().strftime('%H:%M')}**")

            st.markdown("### INTERPRETABILITY")
            
            if not exp_df.empty:
                with st.container(border=True):
                    st.markdown(f"#### Risk Drivers (Active Model: {MODEL_LABELS.get(st.session_state.model_version)})")  
                    
                    if not exp_df.empty and 'Error' not in exp_df.columns:
                        base = alt.Chart(exp_df.head(10)).encode(
                            x=alt.X('Contribution:Q', title="Impact Score"),
                            y=alt.Y('Feature:N', sort='-x', title=None)
                        )
                        
                        bars = base.mark_bar().encode(
                            color=alt.condition(
                                alt.datum.Contribution > 0, 
                                alt.value("#dc2626"), # Red for positive impact (risk)
                                alt.value("#059669")  # Green for negative impact (protective)
                            ),
                            tooltip=['Feature', alt.Tooltip('Contribution:Q', format='.2f')]
                        )
                        
                        # Add a zero line
                        rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='#1e293b', size=1.5).encode(x='x')
                        
                        chart = (bars + rule).properties(height=350)
                        st.altair_chart(chart, use_container_width=True)
                        
                        st.markdown("#### KEY INFLUENCERS")
                        r_col, p_col = st.columns(2)
                        
                        with r_col:
                            st.markdown("**RISK FACTORS (INCREASES PROBABILITY)**")
                            risk_factors = exp_df[exp_df['Contribution'] > 0].head(3)
                            for _, row in risk_factors.iterrows():
                                st.markdown(
                                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                                    f'<span class="risk-badge risk-high">{row["Feature"].upper()}</span>'
                                    f'<span style="font-weight:600; color:#dc2626;">+{row["Contribution"]:.2f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        
                        with p_col:
                            st.markdown("**PROTECTIVE FACTORS (DECREASES PROBABILITY)**")
                            prot_factors = exp_df[exp_df['Contribution'] < 0].sort_values('Contribution').head(3)
                            for _, row in prot_factors.iterrows():
                                st.markdown(
                                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                                    f'<span class="risk-badge risk-low">{row["Feature"].upper()}</span>'
                                    f'<span style="font-weight:600; color:#059669;">{row["Contribution"]:.2f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                        st.info("💡 **Clinical Note:** Positive values indicate features that contribute to readmission risk, while negative values represent protective medical factors.")
                    else:
                        err_msg = exp_df['Error'].iloc[0] if 'Error' in exp_df.columns else "Unknown error"
                        st.error(f"Explanation Unavailable: {err_msg}")

            # Reporting
            
            with st.container(border=True):
                st.markdown("#### ⚕️ FDA POLYPHARMACY SAFETY CHECK")
                
                if interaction_alerts:
                    for alert in interaction_alerts:
                        st.markdown(
                            f"<div style='padding:10px; border-radius:5px; border-left:5px solid {alert['color']}; background-color:#f8fafc; margin-bottom:10px;'>"
                            f"<strong style='color:{alert['color']};'>{alert['level']}</strong><br/>"
                            f"<span style='color:#334155;'>{alert['message']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.success("No drug interactions detected for this patient's current medication profile.")

            with st.container(border=True):
                st.markdown("#### CASE HISTORY")

                # New Intervention Form
                with st.expander("LOG NOTES", expanded=True):

                    with st.form(key=f'intervention_form_{patient_id}'):
                        col_form1, col_form2 = st.columns([1, 1])
                        with col_form1:
                            action_type = st.selectbox(
                                "TYPE", 
                                ["Care Coordination", "Medication Reconciliation", "Home Health Visit", "Specialist Referral", "Discharge Planning"],
                                key='action_select'
                            )
                        with col_form2:
                            clinician = st.text_input("PROVIDER", value=st.session_state.user, disabled=True)
                        
                        notes_text = st.text_area("NOTES", placeholder="Enter notes...", height=100)

                        submit_log = st.form_submit_button("SAVE", use_container_width=True)

                        if submit_log:
                            # Log to Database
                            success = db.log_intervention(patient_id, action_type, st.session_state.user, notes_text)
                            if success:
                                st.success("Saved.")

                                st.rerun() # Rerun to show in history immediately
                
                st.markdown("##### HISTORY LOG")
                if not patient_history.empty:
                    st.dataframe(
                        patient_history[['timestamp', 'action_type', 'notes', 'clinician']],
                        column_config={
                            "timestamp": "Time",
                            "action_type": "Action",
                            "notes": "Clinical Notes",
                            "clinician": "Provider"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No prior interventions recorded for this patient.")

            # Discharge Plan
            st.markdown("### 📋 DISCHARGE PLAN")
            with st.container(border=True):
                st.caption(
                    "Rule based discharge recommendation derived from this patient's "
                    "risk score and feature drivers. Risk is banded (LOW / MODERATE / HIGH) "
                    "- raw probabilities are not presented as precise clinical estimates."
                )

                # Auto-clear plan when the clinician switches to a different patient
                if st.session_state.get("discharge_plan_patient_id") != patient_id:
                    st.session_state.discharge_plan_text = None
                    st.session_state["discharge_plan_patient_id"] = patient_id

                if not exp_df.empty and "Error" not in exp_df.columns:
                    dp_top_features = (
                        exp_df.sort_values("Contribution", ascending=False)["Feature"]
                        .tolist()
                    )
                else:
                    dp_top_features = ["risk_probability"]

                dp_col, _ = st.columns([1, 3])
                with dp_col:
                    dp_btn = st.button(
                        "⚕️ Generate Discharge Plan",
                        type="primary",
                        use_container_width=True,
                        key="dp_generate_btn",
                    )

                if dp_btn:
                    plan_text = generate_discharge_plan(
                        patient_row  = patient_row.to_dict(),
                        risk_score   = float(risk_prob),
                        top_features = dp_top_features,
                        model_id     = st.session_state.model_version,
                        interaction_alerts=interaction_alerts
                    )
                    st.session_state.discharge_plan_text = plan_text
                    st.session_state["discharge_plan_patient_id"] = patient_id
                    db.log_audit(
                        st.session_state.user,
                        "DISCHARGE_PLAN_GENERATED",
                        f"patient={patient_id} model={st.session_state.model_version}",
                    )

                if st.session_state.discharge_plan_text:
                    with st.container(border=True):
                        st.markdown(st.session_state.discharge_plan_text)

                    # Patient PDF download
                    st.markdown("#### 📥 Patient Copy")
                    st.caption(
                        "Download a plain-English letter for the patient with personalised "
                        "diet, exercise, medication, and 'when to return' advice."
                    )
                    try:
                        pdf_buf = generate_patient_discharge_pdf(
                            patient_id   = str(patient_id),
                            patient_row  = patient_row.to_dict(),
                            risk_score   = float(risk_prob),
                            top_features = dp_top_features,
                            model_id     = st.session_state.model_version,
                            interaction_alerts = interaction_alerts
                        )
                        from datetime import datetime
                        st.download_button(
                            label            = "📄 Download Patient Discharge Letter (PDF)",
                            data             = pdf_buf,
                            file_name        = f"discharge_letter_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime             = "application/pdf",
                            use_container_width = True,
                            type             = "primary",
                        )
                    except Exception as pdf_err:
                        st.warning(f"⚠️ Could not generate patient PDF: {pdf_err}")

                    if st.button("🗑 Clear Plan", key="dp_clear_btn"):
                        st.session_state.discharge_plan_text = None
                        st.rerun()
        
        elif st.session_state.predictions is None:
            st.info("Ready: Run analysis to start.")
        
        else:
            st.info("Ready: Select a patient from the 'PRIORITIZATION QUEUE' tab to view analysis.")

# Individual Risk Predictor
with tab4:
    st.markdown("### INDIVIDUAL RISK PREDICTOR")

    manifest = load_manifest(MANIFEST_PATH)

    if manifest is None:
        st.warning(
            "Cannot load `clinical_models/feature_manifest.json`. "
            "Ensure the file exists and is valid JSON."
        )
    else:
        risk_pipeline = load_model(manifest["model_path"])

        if risk_pipeline is None:
            st.warning(
                "The readmission risk model could not be loaded. "
                "Ensure `clinical_models/final_model/lr_classweight_w7_final.joblib` is present and intact."
            )
        else:
            # Compact Header Row
            hdr_col1, hdr_col2 = st.columns([3, 1])
            with hdr_col1:
                 st.caption("Enter patient clinical values below to compute a real-time hospital readmission risk score.")
            with hdr_col2:
                if st.button("↺ Reset Form", use_container_width=True):
                    st.session_state.risk_reset_key += 1
                    st.session_state.risk_result = None
                    st.rerun()

            # Dynamic input form
            with st.container(border=True):
                with st.form(key="patient_risk_form", border=False):
                    patient_df = render_patient_form(manifest, reset_key=st.session_state.risk_reset_key)
                    submitted = st.form_submit_button(
                        "RUN CLINICAL RISK ANALYSIS",
                        type="primary",
                        use_container_width=True
                    )
                
                if submitted:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            
                            try:
                                expected_cols = list(risk_pipeline.feature_names_in_)
                                for col in expected_cols:
                                    if col not in patient_df.columns:
                                        if col == 'num_lab_procedures':
                                            patient_df[col] = 44.0
                                        elif col == 'num_procedures':
                                            patient_df[col] = 1.0
                                        elif col == 'number_diagnoses':
                                            patient_df[col] = 9.0
                                        elif col.startswith('diag_'):
                                            patient_df[col] = "250.0"
                                        elif col in ['race', 'gender']:
                                            patient_df[col] = "Unknown"
                                        elif col == 'admission_source_id':
                                            patient_df[col] = "7"
                                        else:
                                            patient_df[col] = "No"
                                patient_df_ordered = patient_df[expected_cols]
                            except AttributeError:
                                patient_df_ordered = patient_df
                                
                            prob = float(risk_pipeline.predict_proba(patient_df_ordered)[0, 1])
                        band, band_color = compute_risk_band(prob, mode=st.session_state.get('op_mode_name', 'best_f1'))
                        interpretation   = interpret_risk(prob, band, mode=st.session_state.get('op_mode_name', 'best_f1'))
                        st.session_state.risk_result = {
                            "patient_df": patient_df,
                            "prob": prob,
                            "band": band,
                            "band_color": band_color,
                            "interpretation": interpretation,
                            "interaction_alerts": check_drug_interactions(patient_df.iloc[0].to_dict())
                        }
                    except Exception as exc:
                        st.error(f"⚠️ Prediction failed: {exc}")
                        st.session_state.risk_result = None

            # Results display
            if st.session_state.risk_result is not None:
                res        = st.session_state.risk_result
                prob       = res["prob"]
                band       = res["band"]
                band_color = res["band_color"]
                interp     = res["interpretation"]
                patient_df = res["patient_df"]

                # Combined Results & Interpretation Container
                with st.container(border=True):
                    st.markdown("### 📊 READMISSION RISK ASSESSMENT")
                    
                    m1, m2, m3 = st.columns([1, 1, 1])
                    with m1:
                        st.metric("Readmission Prob.", f"{prob:.1%}")
                    with m2:
                        st.metric("Risk Level", band.upper())
                    with m3:
                        st.markdown("**Risk Status**")
                        st.markdown(
                            f'<div style="margin-top:5px;">'
                            f'<span class="risk-badge" style="background-color:{band_color}; '
                            f'color:#ffffff; border:none; border-radius:4px; padding:6px 16px; font-weight:700;">'
                            f'{band.upper()} RISK</span></div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    st.markdown("#### CLINICAL INTERPRETATION")
                    st.info(interp)

                # Polypharmacy check for individual risk prediction
                with st.container(border=True):
                    st.markdown("#### ⚕️ FDA POLYPHARMACY SAFETY CHECK")
                    st.caption("Real time drug-drug interaction scanning via diagnostic data.")
                    
                    if res["interaction_alerts"]:
                        for alert in res["interaction_alerts"]:
                            st.markdown(
                                f"<div style='padding:10px; border-radius:5px; border-left:5px solid {alert['color']}; background-color:#f8fafc; margin-bottom:10px;'>"
                                f"<strong style='color:{alert['color']};'>{alert['level']}</strong><br/>"
                                f"<span style='color:#334155;'>{alert['message']}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.success("No drug interactions detected for this patient's current medication profile.")

                # Patient Input Summary card
                with st.container(border=True):
                    st.markdown("#### PATIENT INPUT SUMMARY")
                    st.caption("Selected values used for this prediction:")
                    summary_rows = []
                    for col in patient_df.columns:
                        spec  = manifest["features"].get(col, {})
                        label = spec.get("description", col.replace("_", " ").title())
                        summary_rows.append({"Feature": label, "Value": str(patient_df[col].iloc[0])})
                    summary_df = pd.DataFrame(summary_rows)
                    half = len(summary_df) // 2
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.dataframe(summary_df.iloc[:half], use_container_width=True, hide_index=True)
                    with sc2:
                        st.dataframe(summary_df.iloc[half:], use_container_width=True, hide_index=True)

        # Academic disclaimer (always visible)
        st.markdown("---")
        st.info(
            "**Academic Use Only** - This tool is a decision support prototype "
            "developed for educational purposes. "
            "It is not a real clinical diagnostic tool and must not be used to "
            "inform actual patient care or treatment decisions."
        )
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85em;'>"
    "Clinical Readmission Risk Dashboard v2.5.0 "
    "| Academic Use Only"
    "</div>",
    unsafe_allow_html=True
)
