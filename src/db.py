import sqlite3
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken

DB_PATH  = Path("data/clinical_db.sqlite")
KEY_PATH = Path("data/.clinical_key")


def get_or_create_key():
    # Generate the encryption key on first run, then reuse it each time
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    with open(KEY_PATH, "rb") as f:
        return f.read()


ENCRYPTION_KEY = get_or_create_key()
cipher_suite   = Fernet(ENCRYPTION_KEY)


@contextmanager
def _db():
    """Open a database connection and guarantee it is closed on exit."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                action_type TEXT    NOT NULL,
                clinician   TEXT    NOT NULL,
                notes       BLOB
            );

            CREATE TABLE IF NOT EXISTS audit (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                user        TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                resource_id TEXT
            );

            CREATE TABLE IF NOT EXISTS prediction_audit (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                encounter_id     TEXT    NOT NULL,
                model_version    TEXT    NOT NULL,
                risk_probability REAL    NOT NULL,
                predicted_label  INTEGER NOT NULL,
                threshold_used   REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS security_audit (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                user         TEXT    NOT NULL,
                event_type   TEXT    NOT NULL,
                resource_id  TEXT,
                outcome      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_security_event
                ON security_audit (event_type);

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_logs_patient
                ON logs (patient_id);

            CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit (user);

            CREATE INDEX IF NOT EXISTS idx_audit_event
                ON audit (event_type);

            CREATE INDEX IF NOT EXISTS idx_pred_encounter
                ON prediction_audit (encounter_id);

            CREATE INDEX IF NOT EXISTS idx_pred_model
                ON prediction_audit (model_version);
        ''')


def log_prediction(encounter_id, model_version, risk_probability, predicted_label, threshold_used):
    log_predictions_batch([(encounter_id, model_version, risk_probability, predicted_label, threshold_used)])


def log_predictions_batch(prediction_list):
    if not prediction_list:
        return

    # Build a list of tuples so we can insert everything in one database call
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = [
        (timestamp, str(p[0]), p[1], float(p[2]), int(p[3]), float(p[4]))
        for p in prediction_list
    ]

    with _db() as conn:
        conn.executemany('''
            INSERT INTO prediction_audit
                (timestamp, encounter_id, model_version, risk_probability, predicted_label, threshold_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)


def log_audit(user, event_type, resource_id=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _db() as conn:
        conn.execute(
            'INSERT INTO audit (timestamp, user, event_type, resource_id) VALUES (?, ?, ?, ?)',
            (timestamp, user, event_type, resource_id)
        )


def log_security_event(user: str, event_type: str, resource_id: str, outcome: str) -> None:
    """Record a security-relevant event (e.g. SHA-256 integrity check) to the security_audit table."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _db() as conn:
        conn.execute(
            'INSERT INTO security_audit (timestamp, user, event_type, resource_id, outcome) '
            'VALUES (?, ?, ?, ?, ?)',
            (timestamp, user, event_type, resource_id, outcome)
        )


def log_intervention(patient_id, action_type, clinician, notes):
    timestamp       = datetime.now().strftime('%Y-%m-%d %H:%M')
    encrypted_notes = cipher_suite.encrypt(notes.encode())

    with _db() as conn:
        conn.execute(
            'INSERT INTO logs (patient_id, timestamp, action_type, clinician, notes) VALUES (?, ?, ?, ?, ?)',
            (str(patient_id), timestamp, action_type, clinician, encrypted_notes)
        )

    log_audit(clinician, "LOG_ENTRY", str(patient_id))
    return True


def get_patient_history(patient_id):
    with _db() as conn:
        df = pd.read_sql_query(
            'SELECT timestamp, action_type, clinician, notes FROM logs WHERE patient_id = ? ORDER BY id DESC',
            conn,
            params=(str(patient_id),)
        )

    def decrypt_note(val):
        try:
            return cipher_suite.decrypt(val).decode()
        except (InvalidToken, Exception):
            return "[ENCRYPTED]"

    if not df.empty:
        df['notes'] = df['notes'].apply(decrypt_note)

    return df
