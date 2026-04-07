from __future__ import annotations

def _band(risk_score: float) -> str:
    from src.risk_calculator import THRESHOLD_HR, THRESHOLD_F1
    if risk_score >= THRESHOLD_HR:
        return "HIGH"
    if risk_score >= THRESHOLD_F1:
        return "MODERATE"
    return "LOW"

# Public entry point

def generate_discharge_plan(
    patient_row: dict,
    risk_score: float,
    top_features: list[str],
    model_id: str,
    interaction_alerts: list[dict] = None,
) -> str:
    """
    Generate a patient-facing discharge plan in plain English.
    Tone, urgency, diet, exercise, and follow-up all adapt to the risk band.
    No external API required.
    """
    def g(key: str, default: str = "unknown") -> str:
        val = patient_row.get(key, default)
        return str(val) if val is not None else default

    band      = _band(risk_score)
    inpatient = int(g("number_inpatient", "0") or 0)
    change    = g("change",  "No")
    insulin   = g("insulin", "No")
    a1c       = g("A1Cresult",     "unknown")
    max_glu   = g("max_glu_serum", "unknown")

    # Opening -- tone varies by band
    if band == "LOW":
        opening = (
            "**Good news!** Based on your recent visit, your chances of needing to "
            "come back to hospital in the next 30 days look **low**. "
            "Your current treatment appears to be working well. "
            "Keep following the advice below to stay that way."
        )
    elif band == "MODERATE":
        opening = (
            "Your results show **some factors** that could increase your chances of "
            "returning to hospital. This is not alarming - but there are a few things "
            "worth paying attention to at home. "
            "Following the steps below will make a real difference."
        )
    else:
        opening = (
            "**Please read this carefully.** Your results suggest a higher chance of "
            "needing to return to hospital in the next month. "
            "This does not mean something will definitely go wrong - but it does mean "
            "the team wants you to be extra careful at home and to get seen quickly "
            "if anything feels off."
        )

    # Drivers in plain English
    _plain = {
        "number_inpatient":   "how often you have been in hospital recently",
        "time_in_hospital":   "how long this stay was",
        "num_medications":    "the number of medications you take",
        "number_emergency":   "past emergency visits",
        "number_outpatient":  "your outpatient visits",
        "num_lab_procedures": "the number of blood tests during this stay",
        "num_procedures":     "the number of procedures performed",
        "number_diagnoses":   "how many health conditions are being managed",
        "a1cresult":          "your long-term blood sugar (HbA1c)",
        "insulin":            "changes to your insulin",
        "change":             "medication changes during this admission",
        "diabetesmed":        "your diabetes medication",
        "max_glu_serum":      "your blood glucose levels during the stay",
    }
    driver_lines = []
    for f in top_features[:3]:
        label = _plain.get(f.replace(" ", "_").lower(), f.replace("_", " "))
        driver_lines.append(f"- {label.capitalize()}")
    drivers_md = "\n".join(driver_lines) if driver_lines else "- Your overall health during this visit"

    # Diet
    diet_tips  = _patient_diet_advice(top_features, patient_row)
    if band == "LOW":
        diet_intro = "Your diet is on the right track. Keep these habits going:"
    elif band == "MODERATE":
        diet_intro = "These changes to what you eat will help keep your blood sugar stable:"
    else:
        diet_intro = "Diet is one of the most important things you can control right now:"
    diet_md = "\n".join(f"- {t}" for t in diet_tips)

    # Exercise
    exercise_tips = _patient_exercise_advice(top_features, patient_row)
    if band == "LOW":
        exercise_intro = "You are in a good position to stay active. Aim for:"
    elif band == "MODERATE":
        exercise_intro = "Regular movement will help your body recover and manage blood sugar:"
    else:
        exercise_intro = "Start very gently - even a little movement each day helps:"
    exercise_md = "\n".join(f"- {t}" for t in exercise_tips)

    # Medications
    alerts = interaction_alerts if interaction_alerts is not None else []
    has_alert = any(bad in a["level"] for a in alerts for bad in ["RISK", "DDI", "ALERT"])
    
    med_section = ""
    if change.lower() in ("ch", "yes", "1") or insulin.lower() in ("up", "down") or has_alert:
        med_lines = [
            "Take **all** your medications exactly as prescribed. "
            "Do not stop any without speaking to your doctor first."
        ]
        if change.lower() in ("ch", "yes", "1"):
            med_lines.append(
                "Your medications changed during this admission. Ask your pharmacist to explain "
                "exactly what changed and why."
            )
        if insulin.lower() in ("up", "down"):
            direction = "increased" if insulin.lower() == "up" else "decreased"
            med_lines.append(
                f"Your insulin was {direction}. Monitor your blood sugar at home and write down "
                "the readings to share at your next appointment."
            )
        
        med_section = (
            "\n\n### Your Medications\n"
            + "\n".join(f"- {m}" for m in med_lines)
        )
        
        if has_alert:
            med_section += "\n\n> **IMPORTANT SAFETY NOTICE:**\n"
            for a in alerts:
                if any(bad in a["level"] for bad in ["RISK", "DDI", "ALERT"]):
                    med_section += f"> - **{a['level']}**: {a['message']}\n"

    # Next appointment
    if band == "HIGH" or inpatient >= 2:
        appt = (
            "You should be seen **within 7 days** of leaving hospital. "
            "If you do not have an appointment yet, call your GP today."
        )
    elif band == "MODERATE" or inpatient >= 1:
        appt = "Book a follow-up **within 14 days**. Contact your GP practice to confirm."
    else:
        appt = (
            "A routine check-up **within 30 days** is all that is needed. "
            "Your GP will likely be in touch."
        )

    # Blood sugar note
    glucose_note = ""
    if a1c in (">8", ">7"):
        glucose_note += (
            "\n\n> **Blood sugar note:** Your HbA1c was above the target range. "
            "Ask about steps to bring it down at your next visit."
        )
    if max_glu in (">200", ">300"):
        glucose_note += (
            "\n> **Glucose alert:** Your blood sugar was high during your stay. "
            "Check it at home every day and record the numbers."
        )
        
    # Clinical Recommendations (Week 9 Robustness)
    rec_md = ""
    if band == "HIGH":
        rec_md = (
            "- **Medication Review:** Required review of all active prescriptions.\n"
            "- **Glucose Monitoring:** Daily glucose monitoring required.\n"
        )
    elif band == "MODERATE":
        rec_md = (
            "- **Patient Education:** Prioritize medication adherence and diet counseling.\n"
            "- **Outpatient Referral:** Endocrine or diabetes specialist referral recommended.\n"
        )
    else:
        rec_md = "- **Standard Pathway:** No additional interventions needed.\n"

    # When to return
    return_signs = [
        "Chest pain or tightness that does not go away",
        "Difficulty breathing while resting",
        "Blood sugar above 15 mmol/L (270 mg/dL) that will not come down",
        "Blood sugar below 4 mmol/L (70 mg/dL) with shaking, sweating or confusion",
        "A high fever (above 38.5 C) for more than 24 hours",
        "Sudden confusion, slurred speech, or difficulty moving - call 999 immediately",
    ]
    if band in ("HIGH", "MODERATE"):
        return_signs.insert(0, "You feel unwell for more than 24 hours and are not improving")
    return_md = "\n".join(f"- {s}" for s in return_signs)

    return_urgency = (
        "Do not wait - call 999 or go to A&E right away if you notice any of these."
        if band == "HIGH"
        else "It is always better to get checked early rather than waiting."
    )

    return f"""\
### {band} RISK

{opening}

---

**What is affecting your result most:**
{drivers_md}

---

### What to Eat and Drink

{diet_intro}

{diet_md}

- Drink at least 6-8 glasses of water every day
- Eat at regular times -- do not skip meals
- Limit sugary drinks, alcohol, and salty or fried food

---

### How to Stay Active

{exercise_intro}

{exercise_md}

> Stop exercising and rest if you feel chest pain, dizziness, or very out of breath.
{med_section}

---

### Your Next Appointment

{appt}
{glucose_note}

---

### Come Back to Hospital or Call Your GP If You Have:

{return_urgency}

{return_md}

---

### Core Recommendations

{rec_md}

---

*This plan is here to help you look after yourself at home. If you are unsure about anything, please ask your care team.*"""

# Top-feature extraction helpers

def get_lr_top_features(pipeline, patient_df, feature_names: list[str], topk: int = 5) -> list[str]:
    """
    Extract top contributing features for an LR pipeline via |coef × scaled value|.
    """
    import numpy as np
    try:
        preprocessor = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get("scaler")
        clf          = pipeline.named_steps.get("classifier")    or pipeline.named_steps.get("model")
        if preprocessor is None or clf is None:
            return feature_names[:topk]
        X_t = preprocessor.transform(patient_df)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        contribs = np.abs(X_t[0] * clf.coef_[0])
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = [f"feature_{i}" for i in range(len(contribs))]
        ranked = sorted(zip(names, contribs), key=lambda x: x[1], reverse=True)
        return [n.split("__")[-1] for n, _ in ranked[:topk]]
    except Exception:
        return feature_names[:topk]

def get_xgb_top_features(pipeline, patient_df, feature_names: list[str], topk: int = 5) -> list[str]:
    """
    Extract top features from an XGBoost pipeline via built-in feature_importances_.
    """
    import numpy as np
    try:
        clf = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")
        if clf is None:
            return feature_names[:topk]
        imps = clf.feature_importances_
        try:
            preprocessor = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get("scaler")
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = feature_names
        ranked = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
        return [n.split("__")[-1] for n, _ in ranked[:topk]]
    except Exception:
        return feature_names[:topk]

# Patient-facing advice builders

# Maps feature names → plain-English diet tips
_DIET_TIPS: dict[str, str] = {
    "number_inpatient": (
        "You have been in hospital more than once recently. "
        "Follow a low-sugar, low-salt diet to keep your blood sugar and blood pressure stable. "
        "Eat plenty of vegetables (spinach, broccoli, carrots), lean proteins (chicken, fish, eggs), "
        "and whole grains (brown rice, oats). Avoid sugary drinks, fried food, and processed snacks."
    ),
    "num_medications": (
        "You are taking several medications. Take them all with water and a small meal to avoid "
        "stomach upset. Avoid grapefruit juice - it can interfere with some medications. "
        "Do not drink alcohol while on diabetes medication."
    ),
    "time_in_hospital": (
        "You have just spent time in hospital. Your body needs energy to recover. "
        "Eat small, regular meals every 3–4 hours. Focus on high-fibre foods (lentils, beans, fruit) "
        "and avoid skipping meals, which can cause your blood sugar to drop."
    ),
    "number_emergency": (
        "You have had emergency visits in the past. Keep healthy snacks nearby (nuts, whole-grain crackers) "
        "so low blood sugar does not catch you off guard. Monitor your blood sugar before and after meals."
    ),
    "num_lab_procedures": (
        "Your blood tests show areas needing attention. Eat foods rich in fibre and low on the glycaemic index "
        "- sweet potato, lentils, oats - to support stable blood sugar throughout the day."
    ),
    "number_outpatient": (
        "Continue with your outpatient care programme. At your appointments, bring a 3-day food diary "
        "to help your care team adjust your meal plan if needed."
    ),
    "A1Cresult": (
        "Your long-term blood sugar (HbA1c) needs attention. Reduce added sugars and refined carbohydrates. "
        "Choose wholemeal bread over white bread, and eat fruit in moderation (2 portions per day). "
        "A consistent meal schedule helps your body manage insulin better."
    ),
    "insulin": (
        "Your insulin dose has changed. Match your meal times to your insulin schedule - "
        "do not eat much later than planned. Count carbohydrates at each meal and aim for "
        "the same amount each day to make your insulin more predictable."
    ),
}

# Maps feature names → plain-English exercise tips
_EXERCISE_TIPS: dict[str, str] = {
    "number_inpatient": (
        "Start slowly after your hospital stay. Begin with 10-minute gentle walks at home and build "
        "up to 30 minutes most days. Chair-based stretching in the morning helps with stiffness."
    ),
    "time_in_hospital": (
        "Rest is important in your first week home. Move around the house regularly to avoid "
        "blood clots - aim to stand up and walk for 5 minutes every hour during the day."
    ),
    "num_medications": (
        "Light to moderate activity (walking, swimming) helps many diabetes medications work better. "
        "Check your blood sugar before and after exercise at first. Stop if you feel dizzy or shaky."
    ),
    "number_emergency": (
        "Exercise regularly to reduce the chance of future emergencies. A 20–30 minute walk after "
        "dinner each evening can significantly improve your blood sugar control."
    ),
    "insulin": (
        "Exercise affects how insulin works. Always carry fast-acting sugar (glucose tablets or juice) "
        "when exercising. Walk, cycle, or swim - avoid very intense exercise until your dose is stable."
    ),
    "A1Cresult": (
        "Regular exercise is one of the most powerful ways to lower your HbA1c. "
        "Aim for at least 150 minutes of moderate activity per week (e.g. 30 minutes, 5 days a week). "
        "Brisk walking, cycling, or swimming all count."
    ),
    "number_outpatient": (
        "Continuing an active lifestyle supports your outpatient care goals. "
        "Consider joining a community walking group or diabetes exercise class for social support."
    ),
}

def _patient_diet_advice(top_features: list[str], patient_row: dict) -> list[str]:
    """Return up to 3 personalised diet tips based on top features."""
    seen = set()
    tips = []
    for feat in top_features:
        feat_key = feat.replace(" ", "_").lower()
        if feat_key in _DIET_TIPS and feat_key not in seen:
            tips.append(_DIET_TIPS[feat_key])
            seen.add(feat_key)
        if len(tips) >= 3:
            break
    if not tips:
        tips.append(
            "Eat a balanced diet with plenty of vegetables, lean protein, and whole grains. "
            "Limit sugar and salt. Drink at least 8 glasses of water a day."
        )
    return tips

def _patient_exercise_advice(top_features: list[str], patient_row: dict) -> list[str]:
    """Return up to 2 personalised exercise tips based on top features."""
    seen = set()
    tips = []
    for feat in top_features:
        feat_key = feat.replace(" ", "_").lower()
        if feat_key in _EXERCISE_TIPS and feat_key not in seen:
            tips.append(_EXERCISE_TIPS[feat_key])
            seen.add(feat_key)
        if len(tips) >= 2:
            break
    if not tips:
        tips.append(
            "Aim for 30 minutes of light activity (such as walking) most days of the week. "
            "Check with your doctor before starting any new exercise programme."
        )
    return tips

def _return_to_hospital_advice(band: str, inpatient: int) -> list[str]:
    """Return a fixed set of 'when to seek help' warning signs."""
    urgent = [
        "Chest pain, tightness, or pressure that does not go away.",
        "Difficulty breathing or shortness of breath at rest.",
        "Blood sugar above 15 mmol/L (270 mg/dL) that does not come down after medication.",
        "Blood sugar below 4 mmol/L (70 mg/dL) with symptoms (shaking, sweating, confusion).",
        "Signs of infection at any wound site: redness, swelling, warmth, or discharge.",
        "High fever (above 38.5°C / 101°F) that does not respond to paracetamol.",
        "Sudden confusion, slurred speech, or difficulty moving - call emergency services immediately.",
    ]
    follow_up = [
        "Your blood sugar stays consistently above 10 mmol/L despite taking your medication.",
        "You feel more tired than usual for more than 2–3 days.",
        "You notice new swelling in your legs or ankles.",
        "You have concerns about your medications or side effects.",
    ]
    if band == "HIGH" or inpatient >= 2:
        follow_up.insert(0, "You feel unwell for more than 24 hours - do not wait.")
    return urgent, follow_up

# PDF renderer

class _PatientLetterPDF:
    """Thin wrapper - builds a patient discharge letter using fpdf."""

    # Colour palette
    _BLUE   = (3,  105, 161)
    _GREEN  = (5,  150, 105)
    _ORANGE = (217, 119,  6)
    _RED    = (220,  38, 38)
    _SLATE  = (30,   41, 59)
    _GREY   = (100, 116, 139)
    _LIGHT  = (241, 245, 249)

    def __init__(self):
        from fpdf import FPDF  # local import to keep module importable w/o fpdf

        class _PDF(FPDF):
            def header(self_):
                self_.set_font("Arial", "B", 16)
                self_.set_text_color(*_PatientLetterPDF._BLUE)
                self_.cell(0, 12, "PATIENT DISCHARGE LETTER", ln=True, align="C")
                self_.set_font("Arial", "", 9)
                self_.set_text_color(*_PatientLetterPDF._GREY)
                from datetime import datetime
                self_.cell(0, 5,
                    f"Issued: {datetime.now().strftime('%d %B %Y')}   |   "
                    "Academic use only - not a clinical document",
                    ln=True, align="C")
                self_.ln(4)
                self_.set_draw_color(*_PatientLetterPDF._BLUE)
                self_.set_line_width(0.6)
                self_.line(10, self_.get_y(), 200, self_.get_y())
                self_.ln(6)

            def footer(self_):
                self_.set_y(-14)
                self_.set_font("Arial", "I", 8)
                self_.set_text_color(*_PatientLetterPDF._GREY)
                self_.cell(0, 6,
                    f"Page {self_.page_no()}  |  This letter is for personal guidance only. "
                    "Always follow advice from your doctor or nurse.",
                    align="C")

        self._pdf = _PDF()
        self._pdf.set_margins(15, 20, 15)
        self._pdf.set_auto_page_break(auto=True, margin=20)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _s(text: str) -> str:
        """Replace characters outside Latin-1 with safe ASCII equivalents."""
        return (
            text
            .replace("\u2014", "--")   # em dash
            .replace("\u2013", "-")    # en dash
            .replace("\u2022", "-")    # bullet
            .replace("\u2192", "->")   # right arrow
            .replace("\u2018", "'")    # left single quote
            .replace("\u2019", "'")    # right single quote
            .replace("\u201c", '"')    # left double quote
            .replace("\u201d", '"')    # right double quote
            .replace("\u2026", "...")  # ellipsis
            .replace("\u00e2\u0080\u0099", "'")  # utf-8 artefact
            # catch-all: drop anything still outside 0-255
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )

    def _section_title(self, title: str, colour=None):
        colour = colour or self._BLUE
        self._pdf.set_font("Arial", "B", 12)
        self._pdf.set_text_color(*colour)
        self._pdf.set_fill_color(*self._LIGHT)
        self._pdf.cell(0, 9, self._s(f"  {title}"), ln=True, fill=True)
        self._pdf.set_text_color(*self._SLATE)
        self._pdf.ln(2)

    def _body(self, text: str, size: int = 10):
        self._pdf.set_font("Arial", "", size)
        self._pdf.set_text_color(*self._SLATE)
        self._pdf.multi_cell(0, 6, self._s(text))
        self._pdf.ln(2)

    def _bullet(self, items: list[str], colour=None):
        colour = colour or self._SLATE
        self._pdf.set_font("Arial", "", 10)
        self._pdf.set_text_color(*colour)
        for item in items:
            self._pdf.set_x(20)
            self._pdf.multi_cell(0, 6, self._s(f"*  {item}"))
            self._pdf.ln(1)
        self._pdf.ln(2)

    # ── public build ─────────────────────────────────────────────────────────

    def build(
        self,
        patient_id: str,
        patient_row: dict,
        risk_score: float,
        top_features: list[str],
        model_id: str,
        interaction_alerts: list[dict] = None,
    ) -> "io.BytesIO":
        import io
        pdf = self._pdf
        pdf.add_page()

        band       = _band(risk_score)
        inpatient  = int(str(patient_row.get("number_inpatient", 0) or 0))
        band_colour = (
            self._RED    if band == "HIGH"     else
            self._ORANGE if band == "MODERATE" else
            self._GREEN
        )

        # Patient greeting
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(*self._SLATE)
        pdf.multi_cell(0, 7,
            f"Dear Patient (ID: {patient_id}),\n\n"
            "Thank you for your stay. This letter summarises the advice your care "
            "team would like you to follow at home to help you recover and stay healthy. "
            "Please read it carefully and keep it somewhere safe."
        )
        pdf.ln(4)

        # Risk summary bar
        self._section_title("YOUR HEALTH TODAY", band_colour)
        band_text = {
            "HIGH":     "Your readmission risk is HIGH. Please follow all advice below closely "
                        "and contact us if you feel unwell.",
            "MODERATE": "Your readmission risk is MODERATE. Following the advice below will "
                        "significantly reduce your chances of returning to hospital.",
            "LOW":      "Your readmission risk is LOW. Keep up the good work - "
                        "the tips below will help you stay that way.",
        }[band]
        self._body(band_text)

        # Risk pill (coloured rectangle)
        pill_x = pdf.get_x()
        pill_y = pdf.get_y()
        pdf.set_fill_color(*band_colour)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.set_x(pill_x)
        label = self._s(f"  {band} RISK  -  Score: {risk_score:.0%}  ")
        pdf.cell(0, 9, label, ln=True, fill=True, align="L")
        pdf.set_text_color(*self._SLATE)
        pdf.ln(4)

        # Top drivers plain text
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*self._GREY)
        pdf.cell(0, 6, "Factors most affecting your risk today:", ln=True)
        pdf.set_font("Arial", "", 10)
        driver_labels = {
            "number_inpatient":   "Previous hospital admissions",
            "time_in_hospital":   "Length of this stay",
            "num_medications":    "Number of medications",
            "number_emergency":   "Previous emergency visits",
            "number_outpatient":  "Outpatient clinic visits",
            "num_lab_procedures": "Number of blood / lab tests",
            "num_procedures":     "Number of procedures performed",
            "number_diagnoses":   "Number of health conditions",
            "A1Cresult":          "Long-term blood sugar (HbA1c)",
            "insulin":            "Insulin dose",
            "change":             "Medication changes",
            "diabetesMed":        "Diabetes medication",
            "max_glu_serum":      "Blood glucose level during stay",
        }
        for feat in top_features[:5]:
            feat_key    = feat.replace(" ", "_").lower()
            plain_label = driver_labels.get(feat_key, feat.replace("_", " ").title())
            pdf.set_x(20)
            pdf.set_text_color(*band_colour)
            pdf.cell(5, 6, "->", ln=False)
            pdf.set_text_color(*self._SLATE)
            pdf.cell(0, 6, plain_label, ln=True)
        pdf.ln(5)

        # Diet
        self._section_title("WHAT TO EAT & DRINK", self._GREEN)
        self._body(
            "A healthy diet is one of the most powerful tools for managing diabetes "
            "and reducing your risk of returning to hospital. Here is advice tailored to your situation:"
        )
        diet_tips = _patient_diet_advice(top_features, patient_row)
        self._bullet(diet_tips, colour=self._SLATE)

        # General quick tips box
        pdf.set_fill_color(240, 253, 244)
        pdf.set_draw_color(*self._GREEN)
        pdf.set_line_width(0.4)
        box_y = pdf.get_y()
        pdf.rect(15, box_y, 180, 28, "D")
        pdf.set_xy(18, box_y + 3)
        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(*self._GREEN)
        pdf.cell(0, 5, "Quick reminders:", ln=True)
        pdf.set_x(18)
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(*self._SLATE)
        pdf.multi_cell(174, 5, self._s(
            "Drink 6-8 glasses of water daily.  |  Eat at regular times each day.  |  "
            "Avoid skipping meals.  |  Limit alcohol.  |  Cut down on sugar and salt."
        ))
        pdf.ln(12)

        # Exercise
        self._section_title("HOW TO STAY ACTIVE", self._BLUE)
        self._body(
            "Regular movement helps control blood sugar and improves your heart health. "
            "Start gently - your body needs time to adjust after a hospital stay."
        )
        exercise_tips = _patient_exercise_advice(top_features, patient_row)
        self._bullet(exercise_tips, colour=self._SLATE)
        self._body(
            "IMPORTANT: Always check with your doctor or nurse before starting a new "
            "exercise routine. Stop immediately if you feel chest pain, dizziness, or very short of breath."
        )

        # Medication reminders
        alerts = interaction_alerts if interaction_alerts is not None else []
        has_alert = any(bad in a["level"] for a in alerts for bad in ["RISK", "DDI", "ALERT"])
        
        change  = str(patient_row.get("change",  "No")).lower()
        insulin = str(patient_row.get("insulin", "No")).lower()
        if change in ("ch", "yes", "1") or insulin in ("up", "down") or has_alert:
            self._section_title("YOUR MEDICATIONS", self._ORANGE)
            med_lines = [
                "Take all medications exactly as prescribed - do not stop or change doses without "
                "speaking to your doctor first.",
            ]
            if change in ("ch", "yes", "1"):
                med_lines.append(
                    "Your medications were changed during this admission. Make sure you understand "
                    "what changed and why - ask your nurse or pharmacist if you are unsure."
                )
            if insulin in ("up", "down"):
                direction = "increased" if insulin == "up" else "decreased"
                med_lines.append(
                    f"Your insulin dose has been {direction}. Monitor your blood sugar at home "
                    "and record the readings to share with your doctor at your next appointment."
                )
            med_lines.append(
                "Keep a list of all your medicines and carry it with you at all times - "
                "especially if you visit another doctor or go to an emergency department."
            )
            self._bullet(med_lines, colour=self._SLATE)
            
            if has_alert:
                self._pdf.ln(2)
                self._pdf.set_font("Arial", "B", 10)
                self._pdf.set_text_color(*self._RED)
                self._pdf.set_x(15)
                self._pdf.cell(0, 6, "IMPORTANT SAFETY NOTICE:", ln=True)
                
                for a in alerts:
                    if any(bad in a["level"] for bad in ["RISK", "DDI", "ALERT"]):
                        clean_lvl = a["level"].replace("🚨", "").replace("⚠️", "").replace("🛑", "").replace("✅", "").strip()
                        clean_msg = a["message"].replace("🚨", "").replace("⚠️", "").replace("🛑", "").replace("✅", "").strip()
                        self._pdf.set_font("Arial", "B", 10)
                        self._pdf.set_text_color(*self._RED)
                        self._pdf.set_x(18)
                        self._pdf.cell(0, 5, self._s(clean_lvl), ln=True)
                        self._pdf.set_font("Arial", "", 9)
                        self._pdf.set_text_color(*self._SLATE)
                        self._pdf.set_x(18)
                        self._pdf.multi_cell(0, 5, self._s(clean_msg))
                        self._pdf.ln(2)

        # When to return
        urgent_signs, followup_signs = _return_to_hospital_advice(band, inpatient)

        self._section_title("CALL 999 / GO TO A&E IMMEDIATELY IF YOU HAVE:", self._RED)
        self._bullet(urgent_signs, colour=self._RED)

        self._section_title("CONTACT YOUR GP OR NURSE IF:", self._ORANGE)
        self._bullet(followup_signs, colour=self._SLATE)

        # Next appointment reminder
        self._section_title("YOUR NEXT APPOINTMENT", self._BLUE)
        if band == "HIGH" or inpatient >= 2:
            appt_msg = (
                "You should be seen within 7 days of leaving hospital. "
                "If you have not received an appointment yet, call your GP today."
            )
        elif band == "MODERATE" or inpatient >= 1:
            appt_msg = (
                "You should be seen within 14 days. "
                "Contact your GP practice to confirm the date and time."
            )
        else:
            appt_msg = (
                "A routine follow-up within 30 days is recommended. "
                "Your GP will be in touch - you can also call them to arrange this."
            )
        self._body(appt_msg)

        # Closing
        pdf.ln(4)
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(*self._GREY)
        pdf.multi_cell(0, 5,
            "This letter was generated by an automated decision-support system for academic "
            "demonstration purposes. It does not replace advice from your clinical team. "
            f"Risk assessment model: {model_id}."
        )

        # Return bytes buffer
        buf = __import__("io").BytesIO()
        result = pdf.output(dest="S")
        if isinstance(result, str):
            buf.write(result.encode("latin-1"))
        else:
            buf.write(result)
        buf.seek(0)
        return buf

def generate_patient_discharge_pdf(
    patient_id: str,
    patient_row: dict,
    risk_score: float,
    top_features: list[str],
    model_id: str,
    interaction_alerts: list[dict] = None,
) -> "io.BytesIO":
    """
    Generate a patient-facing discharge letter as a PDF.

    Written in plain English with personalised:
      - Diet and nutrition advice
      - Exercise recommendations
      - Medication adherence reminders
      - 'When to return to hospital' red-flag list
      - Next appointment guidance

    All advice is derived from the top risk features and patient values -
    no external API required.

    Parameters
    ----------
    patient_id    : str   Identifier shown in the letter header.
    patient_row   : dict  Patient feature values.
    risk_score    : float Raw probability from the active model (0–1).
    top_features  : list  Feature names ranked by importance (up to 5 used).
    model_id      : str   Active model identifier (shown in footnote).

    Returns
    -------
    io.BytesIO  PDF byte stream ready for st.download_button().
    """
    return _PatientLetterPDF().build(
        patient_id   = patient_id,
        patient_row  = patient_row,
        risk_score   = risk_score,
        top_features = top_features,
        model_id     = model_id,
        interaction_alerts = interaction_alerts,
    )

