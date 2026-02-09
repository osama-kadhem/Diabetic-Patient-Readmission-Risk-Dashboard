import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
import os

DB_PATH = Path("data/clinical_db.sqlite")
KEY_PATH = Path("data/.clinical_key")

def get_or_create_key():
    # get key
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    with open(KEY_PATH, "rb") as f:
        return f.read()

ENCRYPTION_KEY = get_or_create_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

def init_db():
    # setup db
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # history table
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
    
    # audit table
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            event_type TEXT,
            resource_id TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_audit(user, event_type, resource_id=None):
    # log event
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO audit (timestamp, user, event_type, resource_id)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, user, event_type, resource_id))
    conn.commit()
    conn.close()

def log_intervention(patient_id, action_type, clinician, notes):
    # save intervention
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # encrypt notes
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
    # get data
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM logs WHERE patient_id = ? ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=(str(patient_id),))
    
    # decrypt
    def decrypt_note(val):
        try:
            return cipher_suite.decrypt(val).decode()
        except:
            return "[ENCRYPTED]"

    if not df.empty:
        df['notes'] = df['notes'].apply(decrypt_note)
    
    conn.close()
    return df
