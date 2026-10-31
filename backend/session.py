import sqlite3, json, os
from datetime import datetime

DB_PATH = os.environ.get("NEUROTRACK_DB", "neurotrack.db")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                finger_score INTEGER,
                spiral_score INTEGER,
                tremor_score INTEGER,
                composite    INTEGER,
                risk_level   TEXT,
                finger_data  TEXT,
                spiral_data  TEXT,
                tremor_data  TEXT
            )
        """)

def save_session(patient_id: str, finger: dict=None, spiral: dict=None, tremor: dict=None) -> dict:
    scores = [s for s in [
        finger.get("overall_score") if finger else None,
        spiral.get("overall_score") if spiral else None,
        tremor.get("overall_score") if tremor else None,
    ] if s is not None]
    composite = int(round(sum(scores)/len(scores))) if scores else None
    risk = _risk_level(composite) if composite is not None else "Unknown"
    now = datetime.utcnow().isoformat()

    with _conn() as c:
        cur = c.execute("""
            INSERT INTO sessions
              (patient_id,created_at,finger_score,spiral_score,tremor_score,
               composite,risk_level,finger_data,spiral_data,tremor_data)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_id, now,
            finger.get("overall_score") if finger else None,
            spiral.get("overall_score") if spiral else None,
            tremor.get("overall_score") if tremor else None,
            composite, risk,
            json.dumps(finger) if finger else None,
            json.dumps(spiral) if spiral else None,
            json.dumps(tremor) if tremor else None,
        ))
        session_id = cur.lastrowid

    return get_session(session_id)   # runs on a fresh connection after commit

def get_session(session_id: int) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return _row_to_dict(row) if row else None

def get_patient_history(patient_id: str, limit: int=10) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY created_at DESC LIMIT ?",
            (patient_id, limit)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

def _row_to_dict(row) -> dict:
    d = dict(row)
    for key in ("finger_data","spiral_data","tremor_data"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d

def _risk_level(score: int) -> str:
    if score >= 80: return "Low"
    if score >= 60: return "Moderate"
    if score >= 40: return "High"
    return "Very High"

init_db()