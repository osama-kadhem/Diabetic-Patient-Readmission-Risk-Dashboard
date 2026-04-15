from fpdf import FPDF
import pandas as pd
from datetime import datetime
import io

class PatientReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(3, 105, 161) # #0369a1
        self.cell(0, 10, 'CLINICAL PATIENT DOSSIER', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.set_text_color(100, 116, 139) # #64748b
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 10, f'Page {self.page_no()} | Confidential Clinical Data', 0, 0, 'C')

def generate_patient_pdf(patient_row, explanation_df=None, history_df=None, user_name="SYSTEM", interaction_alerts=None):
    pdf = PatientReport()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(248, 250, 252) # light background
    pdf.cell(0, 10, f'Patient: {patient_row["patient_id"]}', 0, 1, 'L', fill=True)
    
    pdf.set_font('Arial', '', 10)
    pdf.ln(5)
    
    # Risk score and band
    risk_prob = patient_row['risk_probability']
    risk_band = patient_row['risk_band']
    priority = patient_row['follow_up_priority']
    
    pdf.set_font('Arial', 'B', 11)
    start_y = pdf.get_y()
    pdf.cell(55, 8, 'Readmission Risk Status:', 0, 0)
    
    # Colour the badge to match the risk band
    if risk_band == 'High':
        r, g, b = 220, 38, 38
    elif risk_band == 'Moderate':
        r, g, b = 217, 119, 6
    else:
        r, g, b = 5, 150, 105

    bar_x = pdf.get_x()
    bar_y = start_y + 2
    bar_w = 60
    bar_h = 4
    
    pdf.set_fill_color(226, 232, 240)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, 'F')
    
    if risk_prob > 0:
        pdf.set_fill_color(r, g, b)
        pdf.rect(bar_x, bar_y, bar_w * risk_prob, bar_h, 'F')
        
    pdf.set_xy(bar_x + bar_w + 5, start_y)
    pdf.set_text_color(r, g, b)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, f'{risk_band.upper()} ({risk_prob:.1%})', 0, 1)
    
    pdf.set_text_color(0, 0, 0)
    pdf.cell(55, 8, 'Follow-up Priority Rank:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'#{priority}', 0, 1)
    
    pdf.ln(10)
    
    # Key clinical figures from this encounter
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Clinical Snapshot', 'B', 1, 'L')
    pdf.ln(2)
    pdf.set_font('Arial', '', 10)
    
    metrics_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures', 'num_procedures']
    for col in metrics_cols:
        if col in patient_row:
            label = col.replace('_', ' ').title()
            val = patient_row[col]
            pdf.cell(60, 8, f'{label}:', 0, 0)
            pdf.cell(0, 8, str(val), 0, 1)
            
    # Static fields always shown
    pdf.cell(60, 8, 'Admission Status:', 0, 0)
    pdf.cell(0, 8, 'Stable', 0, 1)
    pdf.cell(60, 8, 'Clinical Provider:', 0, 0)
    pdf.cell(0, 8, user_name, 0, 1)
            
    pdf.ln(10)
    
    # Diverging bar chart of top feature contributions
    if explanation_df is not None and not explanation_df.empty and 'Contribution' in explanation_df.columns:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Interpretability: Key Risk Drivers', 'B', 1, 'L')
        pdf.ln(3)
        
        pdf.set_font('Arial', 'B', 9)
        pdf.set_x(10)
        pdf.cell(75, 6, "Feature", 0, 0, 'R')
        pdf.cell(55, 6, "", 0, 0, 'C')
        pdf.cell(10, 6, "0", 0, 0, 'C')
        pdf.cell(40, 6, "Impact", 0, 1, 'L')
        
        max_val = explanation_df['Contribution'].abs().max()
        if max_val == 0: max_val = 1
        
        center_x = 145
        max_bar_w = 40
        
        top_features = pd.concat([
            explanation_df[explanation_df['Contribution'] > 0].head(4),
            explanation_df[explanation_df['Contribution'] < 0].sort_values('Contribution').head(4)
        ]).sort_values('Contribution', ascending=False)
        
        start_y = pdf.get_y()
        pdf.set_font('Arial', '', 9)
        for _, row in top_features.iterrows():
            feat = row['Feature'].upper()
            val = row['Contribution']
            bar_len = (abs(val) / max_val) * max_bar_w
            
            y = pdf.get_y()
            pdf.set_text_color(100, 116, 139)
            pdf.set_x(10)
            pdf.cell(75, 6, feat, 0, 0, 'R')
            
            if val > 0:
                pdf.set_fill_color(220, 38, 38)
                pdf.rect(center_x, y + 1.5, bar_len, 3, 'F')
                pdf.set_xy(center_x + bar_len + 2, y)
                pdf.set_text_color(220, 38, 38)
                pdf.cell(20, 6, f"+{val:.2f}", 0, 1, 'L')
            else:
                pdf.set_fill_color(5, 150, 105)
                pdf.rect(center_x - bar_len, y + 1.5, bar_len, 3, 'F')
                pdf.set_xy(center_x - bar_len - 17, y)
                pdf.set_text_color(5, 150, 105)
                pdf.cell(15, 6, f"{val:.2f}", 0, 0, 'R')
                pdf.set_y(y + 6)
                
        end_y = pdf.get_y()
        pdf.set_draw_color(30, 41, 59)
        pdf.line(center_x, start_y, center_x, end_y)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

    # Polypharmacy safety section — data passed directly from the UI
    alerts = interaction_alerts or []
    
    if alerts:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, 'FDA Polypharmacy Safety Check', 'B', 1, 'L')
        pdf.ln(4)
        for alert in alerts:
            # Colour varies by severity level
            if "SEVERE" in alert["level"]:
                pdf.set_text_color(159, 18, 57)
            elif "HIGH" in alert["level"]:
                pdf.set_text_color(220, 38, 38)
            elif "MODERATE" in alert["level"]:
                pdf.set_text_color(217, 119, 6)
            else:
                pdf.set_text_color(5, 150, 105)
            
            # Strip emojis for latin-1 PDF compatibility
            clean_lvl = alert["level"].replace("🚨", "").replace("⚠️", "").replace("🛑", "").replace("✅", "").strip()
            clean_msg = alert["message"].replace("🚨", "").replace("⚠️", "").replace("🛑", "").replace("✅", "").strip()
            
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, clean_lvl, 0, 1, 'L')
            pdf.set_text_color(51, 65, 85)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 5, clean_msg)
            pdf.ln(3)

    # Chronological log of clinician interventions
    if history_df is not None and not history_df.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, 'Intervention History', 'B', 1, 'L')
        pdf.ln(4)
        
        for _, row in history_df.iterrows():
            ts = str(row["timestamp"])[:16]
            action = str(row["action_type"])
            
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, f"[{ts}] {action}", 0, 1, 'L')
            
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 5, f"Provider: {row['clinician']}", 0, 1, 'L')
            
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 5, f"Notes: {row['notes']}")
            pdf.ln(3)
            
            pdf.set_draw_color(226, 232, 240)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

    buffer = io.BytesIO()
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        # fpdf returns a latin-1 encoded string in older versions
        buffer.write(pdf_output.encode('latin-1'))
    else:
        buffer.write(pdf_output)
    buffer.seek(0)
    return buffer
