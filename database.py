import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "driftshield.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inference_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    features_json TEXT NOT NULL,
    prediction REAL,
    label REAL,
    label_arrived_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_baseline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('numerical', 'categorical')),
    baseline_json TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(feature_name)
);

CREATE TABLE IF NOT EXISTS drift_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    test_name TEXT NOT NULL,
    statistic REAL,
    p_value REAL,
    corrected_p_value REAL,
    significant INTEGER NOT NULL DEFAULT 0,
    drift_type TEXT NOT NULL CHECK (drift_type IN ('covariate', 'concept')),
    window_end_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    label REAL NOT NULL,
    arrived_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES inference_logs(request_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_logs_created
    ON inference_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_results_created
    ON drift_test_results(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_results_feature
    ON drift_test_results(feature_name);
CREATE INDEX IF NOT EXISTS idx_labels_request
    ON labels(request_id);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def insert_inference(
    request_id: str,
    features: dict,
    prediction: float,
    label: Optional[float] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO inference_logs
               (request_id, features_json, prediction, label, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (request_id, json.dumps(features), prediction, label, now),
        )


def get_recent_inferences(
    limit: int = 1000, offset: int = 0
) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM inference_logs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()


def get_inference_window(
    since: str, until: str
) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM inference_logs WHERE created_at BETWEEN ? AND ? ORDER BY created_at ASC",
            (since, until),
        ).fetchall()


def upsert_reference_baseline(
    feature_name: str,
    feature_type: str,
    baseline_data: dict,
    n_samples: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO reference_baseline
               (feature_name, feature_type, baseline_json, n_samples, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (feature_name, feature_type, json.dumps(baseline_data), n_samples, now),
        )


def get_reference_baselines() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM reference_baseline"
        ).fetchall()


def insert_drift_result(
    feature_name: str,
    test_name: str,
    statistic: float,
    p_value: float,
    corrected_p_value: float,
    significant: bool,
    drift_type: str,
    window_end_at: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO drift_test_results
               (feature_name, test_name, statistic, p_value,
                corrected_p_value, significant, drift_type,
                window_end_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (feature_name, test_name, statistic, p_value,
             corrected_p_value, int(significant), drift_type,
             window_end_at, now),
        )


def get_recent_drift_results(limit: int = 200) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM drift_test_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_latest_drift_results() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """SELECT d.* FROM drift_test_results d
               INNER JOIN (
                   SELECT feature_name, MAX(created_at) AS max_created
                   FROM drift_test_results GROUP BY feature_name
               ) latest ON d.feature_name = latest.feature_name
               AND d.created_at = latest.max_created"""
        ).fetchall()


def insert_label(request_id: str, label: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO labels (request_id, label, arrived_at) VALUES (?, ?, ?)",
            (request_id, label, now),
        )
        conn.execute(
            "UPDATE inference_logs SET label = ?, label_arrived_at = ? WHERE request_id = ?",
            (label, now, request_id),
        )
