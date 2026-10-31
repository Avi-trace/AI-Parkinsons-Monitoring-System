"""
Persistence layer.

Everything the API needs from storage goes through `Repository`, so moving to
Postgres / Supabase later means writing one new class with the same methods —
no route or analyzer changes. SQLite-specific bits to translate for Postgres:
AUTOINCREMENT (-> IDENTITY), INSERT OR IGNORE/REPLACE (-> ON CONFLICT), the PRAGMAs,
and "?" placeholders (-> %s with psycopg).

Tables
  patients      one row per patient (no names / contact details stored)
  sessions      one assessment visit: medication state, device, composite index
  test_results  one row per analysed test; created by the analysis endpoints
                (session_id NULL) and attached to a session when it is saved.
                Scores are always computed server-side — clients never send them.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 2


class ConflictError(Exception):
    """A concurrent request changed the rows this one depends on (-> HTTP 409)."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS patients (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    dominant_hand  TEXT CHECK (dominant_hand IN ('left', 'right') OR dominant_hand IS NULL)
);
CREATE TABLE IF NOT EXISTS sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          TEXT NOT NULL REFERENCES patients(id),
    created_at          TEXT NOT NULL,
    medication_state    TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (medication_state IN ('on', 'off', 'unknown')),
    minutes_since_dose  INTEGER,
    device              TEXT,
    algorithm_version   TEXT NOT NULL,
    screening_index     INTEGER,
    screening_label     TEXT,
    idempotency_key     TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_sessions_patient_created ON sessions (patient_id, created_at);
CREATE TABLE IF NOT EXISTS test_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id         TEXT NOT NULL REFERENCES patients(id),
    session_id         INTEGER REFERENCES sessions(id),
    created_at         TEXT NOT NULL,
    test_type          TEXT NOT NULL CHECK (test_type IN ('finger_tap', 'tremor', 'spiral')),
    side               TEXT,
    mode               TEXT,
    score              INTEGER,
    severity           TEXT,
    algorithm_version  TEXT NOT NULL,
    summary_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_results_session ON test_results (session_id);
CREATE INDEX IF NOT EXISTS ix_results_patient ON test_results (patient_id, created_at);
"""

# Large arrays are only needed to draw the result once; they are not stored.
_DROP_KEYS = {"raw_frames", "tap_events", "fft_spectrum", "left_spectrum", "right_spectrum"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in _DROP_KEYS}


class Repository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False) if db_path == ":memory:" else None
        self.init_schema()

    # ── connection handling ──────────────────────────────────────────────────
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._memory_conn or sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._memory_conn is None:
                conn.close()

    def init_schema(self) -> None:
        with self._conn() as c:
            if self._memory_conn is None:
                c.execute("PRAGMA journal_mode = WAL")
            c.executescript(SCHEMA)
            c.execute("INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                      (str(SCHEMA_VERSION),))

    # ── patients ─────────────────────────────────────────────────────────────
    def ensure_patient(self, c: sqlite3.Connection, patient_id: str) -> None:
        c.execute("INSERT OR IGNORE INTO patients (id, created_at) VALUES (?, ?)", (patient_id, _now()))

    def upsert_patient(self, patient_id: str, dominant_hand: str | None) -> dict:
        with self._conn() as c:
            self.ensure_patient(c, patient_id)
            c.execute("UPDATE patients SET dominant_hand = ? WHERE id = ?", (dominant_hand, patient_id))
            return dict(c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone())

    def get_patient(self, patient_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
            return dict(row) if row else None

    # ── test results ─────────────────────────────────────────────────────────
    def add_result(self, *, patient_id: str, test_type: str, side: str | None, mode: str | None,
                   score: int | None, severity: str | None, algorithm_version: str, result: dict) -> int:
        with self._conn() as c:
            self.ensure_patient(c, patient_id)
            cur = c.execute(
                """INSERT INTO test_results (patient_id, created_at, test_type, side, mode, score, severity,
                                             algorithm_version, summary_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, _now(), test_type, side, mode, score, severity, algorithm_version,
                 json.dumps(summarize(result))),
            )
            return int(cur.lastrowid)

    def get_results(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        with self._conn() as c:
            q = f"SELECT * FROM test_results WHERE id IN ({','.join('?' * len(ids))})"
            return [self._result_row(r) for r in c.execute(q, ids).fetchall()]

    @staticmethod
    def _result_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json"))
        return d

    # ── sessions ─────────────────────────────────────────────────────────────
    def find_session_by_key(self, key: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT id FROM sessions WHERE idempotency_key = ?", (key,)).fetchone()
        return self.get_session(row["id"]) if row else None

    def create_session(self, *, patient_id: str, medication_state: str, minutes_since_dose: int | None,
                       device: str | None, algorithm_version: str, screening_index: int | None,
                       screening_label: str | None, idempotency_key: str | None, result_ids: list[int]) -> int:
        with self._conn() as c:
            self.ensure_patient(c, patient_id)
            cur = c.execute(
                """INSERT INTO sessions (patient_id, created_at, medication_state, minutes_since_dose, device,
                                         algorithm_version, screening_index, screening_label, idempotency_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, _now(), medication_state, minutes_since_dose, device, algorithm_version,
                 screening_index, screening_label, idempotency_key),
            )
            sid = int(cur.lastrowid)
            if result_ids:
                cur = c.execute(
                    f"UPDATE test_results SET session_id = ? "
                    f"WHERE session_id IS NULL AND patient_id = ? AND id IN ({','.join('?' * len(result_ids))})",
                    [sid, patient_id, *result_ids],
                )
                if cur.rowcount != len(result_ids):   # attached concurrently -> roll back everything
                    raise ConflictError("one or more results were attached to another session")
            return sid

    def get_session(self, session_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None
            results = c.execute("SELECT * FROM test_results WHERE session_id = ? ORDER BY id",
                                (session_id,)).fetchall()
            return {**dict(row), "results": [self._result_row(r) for r in results]}

    def patient_history(self, patient_id: str, limit: int) -> list[dict]:
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM sessions WHERE patient_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (patient_id, limit)).fetchall()]
        return [s for s in (self.get_session(i) for i in ids) if s]
