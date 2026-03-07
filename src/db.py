import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
import os

DB_PATH  = Path("data/clinical_db.sqlite")
KEY_PATH = Path("data/.clinical_key")

def get_or_create_key():
    # generate and persist an encryption key on first run
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    with open(KEY_PATH, "rb") as f:
        return f.read()

ENCRYPTION_KEY = get_or_create_key()
cipher_suite   = Fernet(ENCRYPTION_KEY)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp TEXT,
            action_type TEXT,
            clinician TEXT,
            notes BLOB
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            event_type TEXT,
            resource_id TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            encounter_id TEXT,
            model_version TEXT,
            risk_probability REAL,
            predicted_label INTEGER,
            threshold_used REAL
        )
    ''')

    conn.commit()
    conn.close()

def log_prediction(encounter_id, model_version, risk_probability, predicted_label, threshold_used):
    log_predictions_batch([(encounter_id, model_version, risk_probability, predicted_label, threshold_used)])

def log_predictions_batch(prediction_list):
    # write a batch of predictions in a single transaction
    if not prediction_list:
        return

    conn      = sqlite3.connect(DB_PATH)
    c         = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = [
        (timestamp, str(p[0]), p[1], float(p[2]), int(p[3]), float(p[4]))
        for p in prediction_list
    ]

    c.executemany('''
        INSERT INTO prediction_audit
        (timestamp, encounter_id, model_version, risk_probability, predicted_label, threshold_used)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data)

    conn.commit()
    conn.close()

def log_audit(user, event_type, resource_id=None):
    conn      = sqlite3.connect(DB_PATH)
    c         = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO audit (timestamp, user, event_type, resource_id)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, user, event_type, resource_id))
    conn.commit()
    conn.close()

def log_intervention(patient_id, action_type, clinician, notes):
    conn      = sqlite3.connect(DB_PATH)
    c         = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # encrypt notes before storing
    encrypted_notes = cipher_suite.encrypt(notes.encode())

    c.execute('''
        INSERT INTO logs (patient_id, timestamp, action_type, clinician, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (str(patient_id), timestamp, action_type, clinician, encrypted_notes))

    conn.commit()
    conn.close()

    log_audit(clinician, "LOG_ENTRY", str(patient_id))
    return True

def get_patient_history(patient_id):
    conn  = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM logs WHERE patient_id = ? ORDER BY id DESC"
    df    = pd.read_sql_query(query, conn, params=(str(patient_id),))

    def decrypt_note(val):
        try:
            return cipher_suite.decrypt(val).decode()
        except:
            return "[ENCRYPTED]"

    if not df.empty:
        df['notes'] = df['notes'].apply(decrypt_note)

    conn.close()
    return df
