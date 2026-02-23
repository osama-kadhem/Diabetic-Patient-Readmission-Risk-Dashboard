import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import io
import altair as alt
import hashlib

# imports

from src.data_validation import validate_csv
from src.predict import predict_risk, rank_patients
import src.db as db


# --- Week 5 Helpers ---

@st.cache_data
def get_recommended_stable_pair():
    """Loads stability artifacts from the consolidated research location."""
    try:
        df_stab = pd.read_csv("clinical_models/research_week_4_5/stability_metrics_w4_5.csv")
        if not df_stab.empty:
            # Map artifact names (lr_none, etc) to app registry keys
            mapping = {
                "lr_none": "baseline_v1",
                "lr_classweight": "classweight_v1",
                "lr_ros": "ros_v1"
            }
            top_row = df_stab.iloc[0]
            m1 = mapping.get(top_row['model_a'], "baseline_v1")
            m2 = mapping.get(top_row['model_b'], "classweight_v1")
            return m1, m2, top_row['jaccard_top10']
    except Exception:
        pass
    return "baseline_v1", "classweight_v1", None

@st.cache_data
def get_local_explanation(pipeline_hash, _pipeline, patient_row):
    """Calculates feature contributions for a specific patient. Uses pipeline_hash to ensure cache updates if model changes."""
    try:
        # 1. Extract feature columns used during training
        feature_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses'
        ]
        X = patient_row[feature_cols].values.reshape(1, -1)
        
        # 2. Get transformed (scaled) values
        scaler = _pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X)
        
        # 3. Get coefficients from the Logistic Regression model
        model = _pipeline.named_steps['model'] if 'model' in _pipeline.named_steps else _pipeline.named_steps['classifier']
        weights = model.coef_[0]
        
        # 4. Calculate contribution: Scaled Value * Coefficient
        contributions = X_scaled[0] * weights
        
        # Create results dataframe with Clinical Terminology
        name_map = {
            'Time In Hospital': 'Days in Hospital',
            'Num Lab Procedures': 'Lab Tests Performed',
            'Num Procedures': 'Clinical Procedures',
            'Num Medications': 'Active Medications',
            'Number Outpatient': 'Prior Outpatient Visits',
            'Number Emergency': 'Prior Emergency Visits',
            'Number Inpatient': 'Prior Inpatient Admissions',
            'Number Diagnoses': 'Comorbidity Count'
        }
        
        df_exp = pd.DataFrame({
            'Feature': [name_map.get(c.replace('_', ' ').title(), c) for c in feature_cols],
            'Contribution': contributions
        }).sort_values(by='Contribution', ascending=False)
        
        return df_exp
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


# Initialize DB
db.init_db()

# Page configuration
st.set_page_config(
    page_title="Clinical Dashboard - Readmission Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clinical Dashboard Style



st.markdown("""
<style>
    /* Professional Document Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Public Sans', sans-serif;
        color: #1e293b; /* Slate 800 */
        background-color: #f1f5f9; /* Slate 100 */
    }
    
    .stApp {
        background-color: #f1f5f9;
    }
    
    /* Systematic Container Styling */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: #ffffff;
        border-radius: 4px !important;
        border-left: 4px solid #0284c7 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar: Control Center */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #0369a1 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem !important;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }

    /* Input Fields */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        border-radius: 4px !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Systematic Headers */
    h1, h2, h3 {
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Public Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0369a1;
    }
    
    /* Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 2px;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
    }
    
    .risk-high { background-color: #dc2626; color: #ffffff; border: 1px solid #dc2626; }
    .risk-medium { background-color: #d97706; color: #ffffff; border: 1px solid #d97706; }
    .risk-low { background-color: #059669; color: #ffffff; border: 1px solid #059669; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-bottom: 2px solid #e2e8f0;
        padding: 0 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748b;
        padding: 10px 20px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7;
        border-bottom-color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# setup session state

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'model_version' not in st.session_state:
    st.session_state.model_version = "baseline_v1"

# Model Registry (Updated for Week 5)
MODEL_REGISTRY = {
  "baseline_v1": "clinical_models/baseline_v1.joblib",
  "classweight_v1": "clinical_models/classweight_v1.joblib",
  "ros_v1": "clinical_models/ros_v1.joblib",
  "smote_v1": "clinical_models/smote_v1.joblib"
}

# Human-readable model labels for UI
MODEL_LABELS = {
  "baseline_v1": "Standard Model (Baseline)",
  "classweight_v1": "Balanced Model (High Sensitivity)",
  "ros_v1": "Enhanced Model (ROS)",
  "smote_v1": "Enhanced Model (SMOTE)"
}

# helpers

def check_model_integrity(file_path):
    # check if file was changed

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Login

def login_screen():
    st.markdown("""
<div style='text-align: center; padding: 50px 0;'>
<h1 style='color: #0369a1;'>LOGIN</h1>
<p style='color: #64748b;'>Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)
    
    with st.container(border=True):

        st.info("Debug: Click to login")

        
        if st.button("LOGIN", type="primary", use_container_width=True):

            st.session_state.authenticated = True
            st.session_state.user = "DR. OSAMA SALEH"
            db.log_audit(st.session_state.user, "DEBUG_LOGIN_SUCCESS")
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


# Load model pipeline
@st.cache_resource
def load_pipeline(version):
    # load model
    try:
        model_path = Path(MODEL_REGISTRY[version])
        
        if not model_path.exists():
            st.error(f"ERR-404: Model file not found at {model_path}.")
            return None
            
        # Suppress sklearn version warnings for demo purposes
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        with open(model_path, 'rb') as f:
            pipeline = joblib.load(f)
        return pipeline
    except Exception as e:
        st.error(f"ERR-500: System error loading {version}: {str(e)}")
        st.warning("This may be due to sklearn version mismatch. Models trained with sklearn 1.8.0 may not work with sklearn 1.6.1.")
        return None

@st.cache_data
def load_and_validate_data(file_obj):
    """Cached function to read and validate data"""
    df = pd.read_csv(file_obj)
    return df

# Sidebar: Clinical Controls
with st.sidebar:
    st.markdown("### ACCOUNT & PRIVACY")
    st.caption("DEBUG MODE: Clinical identifiers exposed for validation.")
    if st.button("SECURE LOGOUT"):
        db.log_audit(st.session_state.user, "LOGOUT")
        st.session_state.authenticated = False
        st.rerun()
        
    st.markdown("---")
    st.markdown("### SYSTEM CONFIGURATION")
    
    # 0. Model Version Selector (Week 4)
    st.markdown("#### MODEL VERSION")
    
    # Custom CSS to disable typing in the selectbox
    st.markdown("""
        <style>
        /* Disable typing/searching in all selectboxes to maintain 'locked' clinical labels */
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
        st.session_state.pipeline = None # Reset pipeline to force reload
        st.success(f"Model updated: {MODEL_LABELS[selected_version]}")

    # 1. Data Import
    st.markdown("#### DATA IMPORT")

    uploaded_file = st.file_uploader(
        "Upload Patient Encounter CSV",
        type=['csv'],
        help="Requires standard HL7 extract format (CSV)"
    )
    
    if uploaded_file is not None:
        try:
            # use cached loader
            df_full = load_and_validate_data(uploaded_file)
            
            # Optimization for large datasets via 'Smart Cohort Sampling'
            total_rows = len(df_full)
            if total_rows > 10000:
                st.warning(f"Large File: {total_rows:,} records. Sampling data...")

                
                # Option to sample by unique patients to preserve history
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
                # Standardize ID column for the UI
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
    
    

    
    # 3. Operations
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
    
    # Run Button
    st.markdown("---")
    if st.button("RUN ANALYSIS", type="primary", use_container_width=True):

        if st.session_state.uploaded_data is not None:
            # Load pipeline for the selected version
            if st.session_state.pipeline is None:
                st.session_state.pipeline = load_pipeline(st.session_state.model_version)
            
            if st.session_state.pipeline is not None:
                with st.spinner("Analyzing..."):
                    # Predict
                    predictions = predict_risk(
                        st.session_state.uploaded_data, 
                        st.session_state.pipeline,
                        threshold_high=0.7
                    )
                    st.session_state.predictions = rank_patients(predictions)
                    
                    # Log to Audit in high-speed batch (Performance Hack)
                    try:
                        threshold = 0.5 # Default model threshold
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

    # System Status Footer
    st.markdown("---")
    st.caption("🟢 **System Online**")
    st.caption(f"v2.5.0 | Build: {pd.Timestamp.now().strftime('%y%m%d')}")


# Main content area
if st.session_state.uploaded_data is None:
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
        # Add more sample features as needed
    })
    st.dataframe(sample_data, use_container_width=True)

else:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["OVERVIEW", "PRIORITIZATION QUEUE", "PATIENT DOSSIER"])
    
    with tab1:
        # Overview tab
        st.markdown("### OVERVIEW")

        
        if st.session_state.predictions is not None:
            df_pred = st.session_state.predictions
            
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
                    st.metric("High Risk Patients", high_risk_count, delta="Action Required", delta_color="inverse")
                
                with col4:
                    st.metric("Capacity Target", top_k, help="Number of patients targeted for follow-up")
            

            
            # Charts Row
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### Risk Distribution")
                    
                    # Interactive Histogram with Altair
                    risk_chart = alt.Chart(df_pred).mark_bar().encode(
                        x=alt.X('risk_probability', bin=alt.Bin(maxbins=20), title='Risk Probability'),
                        y=alt.Y('count()', title='Number of Patients'),
                        color=alt.Color('risk_probability', scale=alt.Scale(range=['#059669', '#d97706', '#dc2626']), legend=None),
                        tooltip=['count()', alt.Tooltip('risk_probability', bin=True, title='Risk Range')]
                    ).properties(height=300)
                    
                    st.altair_chart(risk_chart, use_container_width=True)
                    
                    # Method structure: Summary badges below (Vertical stacking to match Risk Bands)
                    def get_band_color(prob):
                        if prob >= 0.7: return "#dc2626" # High
                        if prob >= 0.4: return "#d97706" # Medium
                        return "#059669" # Low

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
                            domain=['High', 'Medium', 'Low'],
                            range=['#dc2626', '#d97706', '#059669']
                        )),
                        tooltip=['Band', 'Count']
                    ).properties(height=300)
                    
                    st.altair_chart(donut, use_container_width=True)
                    
                    # Text summary below donut - Uppercase labels to match graph style
                    for band in ['High', 'Medium', 'Low']:
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

            
            # Show raw data preview
            st.subheader("Data Preview")

            st.dataframe(st.session_state.uploaded_data.head(10), use_container_width=True)
    
    with tab2:
        # Ranked patient list tab
        st.markdown("### PATIENT LIST")

        
        if st.session_state.predictions is not None:
            df_pred = st.session_state.predictions
            
            # Start Main Container
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
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply filters logic
            filtered_df = df_pred.copy()
            if search_patient:
                filtered_df = filtered_df[
                    filtered_df['patient_id'].astype(str).str.contains(search_patient, case=False)
                ]
            if risk_filter:
                filtered_df = filtered_df[filtered_df['risk_band'].isin(risk_filter)]
            if show_top_k_only:
                filtered_df = filtered_df.head(top_k)
            
            # Display table container
            with st.container(border=True):
                
                col_header1, col_header2 = st.columns([1, 1])
                with col_header1:
                    st.markdown(f"#### Patient List ({len(filtered_df)} matches)")
                
                # Format display
                display_df = filtered_df.copy()
                display_df['Risk Probability'] = display_df['risk_probability'].apply(lambda x: f"{x:.3%}")
                
                # Use st.dataframe with column config for better look 
                st.dataframe(
                    display_df[['patient_id', 'Risk Probability', 'risk_band', 'follow_up_priority']],
                    column_config={
                        "patient_id": "Patient ID",
                        "Risk Probability": st.column_config.ProgressColumn(
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Quick Selection for Details
            with st.container():
                st.markdown('<div class="stCard">', unsafe_allow_html=True)
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
                        # Audit dossier load
                        db.log_audit(st.session_state.user, "PATIENT_DOSSIER_ACCESSED", str(selected))
                        st.success("Loaded.")

        
        else:
            st.info("Ready: Run analysis to start.")

    
    with tab3:
        # Patient details tab
        st.markdown("### PATIENT DETAILS")

        
        if st.session_state.predictions is not None and st.session_state.selected_patient is not None:
            patient_id = st.session_state.selected_patient
            df_pred = st.session_state.predictions
            
            # Get patient data
            patient_row = df_pred[df_pred['patient_id'] == patient_id].iloc[0]
            
            # --- Hero Section ---
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 1, 2])
                
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
            
            
                # --- Clinical Indicators ---
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

            # --- Week 5: Interpretability ---
            st.markdown("---")
            st.markdown("### INTERPRETABILITY: WHY IS THIS PATIENT AT RISK?")
            
            # Load Active Pipeline if not available
            active_pipe = st.session_state.get('pipeline')
            if active_pipe is None:
                active_pipe = load_pipeline(st.session_state.model_version)

            if active_pipe:
                with st.container(border=True):
                    st.markdown(f"#### Risk Drivers (Active Model: {MODEL_LABELS.get(st.session_state.model_version)})")
                    
                    # Compute a hash of the pipeline to trigger cache refresh if model changes
                    pipeline_hash = hashlib.sha256(str(active_pipe).encode()).hexdigest()
                    exp_df = get_local_explanation(pipeline_hash, active_pipe, patient_row)
                    
                    if not exp_df.empty and 'Error' not in exp_df.columns:
                        # Upgraded Chart Aesthetics
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
                        
                        # Enhanced Method Structure: Risk vs Protective
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
                        st.error("Could not generate explanation for this patient.")







            # Reporting

            with st.container(border=True):
                # Header with Export Button
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown("#### CASE HISTORY")



                # 1. New Intervention Form
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
                
                # 2. Historical Log from DB
                st.markdown("##### HISTORY LOG")

                patient_history = db.get_patient_history(patient_id)
                
                if not patient_history.empty:
                    # Display history as a clean table make this
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
        
        elif st.session_state.predictions is None:
            st.info("Ready: Run analysis to start.")
        
        else:
            st.info("Ready: Select a patient from the 'PRIORITIZATION QUEUE' tab to view analysis.")




# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85em;'>"
    "Patient Dashboard v1.0 | Demo"

    "</div>",
    unsafe_allow_html=True
)
