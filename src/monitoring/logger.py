"""SQLite audit log for every prediction served.

Schema is intentionally denormalised - one row per call, one optional
drug_b column (null for single-drug calls). Makes ad-hoc queries
(volume, latency, novel-pair rate) trivial.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    endpoint     TEXT    NOT NULL,
    cosmic_id    INTEGER NOT NULL,
    drug_name    TEXT    NOT NULL,
    drug_b       TEXT,
    predicted    REAL,
    is_novel     INTEGER,
    latency_ms   REAL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts       ON predictions (ts);
CREATE INDEX IF NOT EXISTS idx_predictions_endpoint ON predictions (endpoint);
CREATE INDEX IF NOT EXISTS idx_predictions_drug     ON predictions (drug_name);
"""


class PredictionLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _write(self, row: tuple) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(ts, endpoint, cosmic_id, drug_name, drug_b, predicted, is_novel, latency_ms) "
                "VALUES (?,?,?,?,?,?,?,?)",
                row,
            )

    def log_single(self, cosmic_id: int, drug_name: str,
                   predicted_ln_ic50: float, is_novel: bool, latency_ms: float) -> None:
        self._write((time.time(), "single", int(cosmic_id), str(drug_name), None,
                     float(predicted_ln_ic50), int(bool(is_novel)), float(latency_ms)))

    def log_combination(self, cosmic_id: int, drug_a: str, drug_b: str,
                        synergy_score: float, latency_ms: float) -> None:
        self._write((time.time(), "combination", int(cosmic_id), str(drug_a), str(drug_b),
                     float(synergy_score), None, float(latency_ms)))

    def log_scan(self, cosmic_id: int, drug_a: str, top_k: int, latency_ms: float) -> None:
        self._write((time.time(), "scan", int(cosmic_id), str(drug_a), None,
                     None, None, float(latency_ms)))

    def recent(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT id, ts, endpoint, cosmic_id, drug_name, drug_b, predicted, is_novel, latency_ms "
                "FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
