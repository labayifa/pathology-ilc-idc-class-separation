"""Single SQLite DB (one file, default results/experiments.db) tracking every
config run's per-epoch steps, checkpoint locations, and final metrics -- one
durable, queryable record instead of scattered per-run files, so progress
survives a Spot-instance reclaim (2-minute notice, no graceful shutdown
guaranteed) and can be inspected without WandB access.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT UNIQUE NOT NULL,
    feature_set TEXT,
    classifier_head TEXT,
    config_json TEXT,
    status TEXT DEFAULT 'running',
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    seed INTEGER, fold INTEGER, epoch INTEGER,
    train_loss REAL, val_auroc REAL, timestamp TEXT,
    UNIQUE(experiment_id, seed, fold, epoch)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    seed INTEGER, fold INTEGER, kind TEXT, path TEXT,
    metric_value REAL, epoch INTEGER, updated_at TEXT,
    UNIQUE(experiment_id, seed, fold, kind)
);

CREATE TABLE IF NOT EXISTS final_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    auroc REAL, auroc_ci_lo REAL, auroc_ci_hi REAL,
    balanced_accuracy REAL, balanced_accuracy_ci_lo REAL, balanced_accuracy_ci_hi REAL,
    brier REAL, brier_ci_lo REAL, brier_ci_hi REAL,
    wandb_url TEXT, timestamp TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # tolerates a concurrent reader while a run is writing
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_experiment(conn: sqlite3.Connection, run_name: str, feature_set: str,
                              classifier_head: str, config: dict) -> int:
    row = conn.execute("SELECT id FROM experiments WHERE run_name = ?", (run_name,)).fetchone()
    if row:
        conn.execute("UPDATE experiments SET status='running', updated_at=? WHERE id=?", (_now(), row[0]))
        return row[0]
    cur = conn.execute(
        "INSERT INTO experiments (run_name, feature_set, classifier_head, config_json, status, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (run_name, feature_set, classifier_head, json.dumps(config), "running", _now(), _now()),
    )
    return cur.lastrowid


def log_step(conn: sqlite3.Connection, experiment_id: int, seed: int, fold: int, epoch: int,
             train_loss: float, val_auroc: float | None) -> None:
    conn.execute(
        "INSERT INTO steps (experiment_id, seed, fold, epoch, train_loss, val_auroc, timestamp) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(experiment_id, seed, fold, epoch) DO UPDATE SET "
        "train_loss=excluded.train_loss, val_auroc=excluded.val_auroc, timestamp=excluded.timestamp",
        (experiment_id, seed, fold, epoch, train_loss, val_auroc, _now()),
    )


def get_last_epoch(conn: sqlite3.Connection, experiment_id: int, seed: int, fold: int) -> int | None:
    row = conn.execute(
        "SELECT MAX(epoch) FROM steps WHERE experiment_id=? AND seed=? AND fold=?",
        (experiment_id, seed, fold),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def upsert_checkpoint(conn: sqlite3.Connection, experiment_id: int, seed: int, fold: int, kind: str,
                       path: str, metric_value: float | None, epoch: int) -> None:
    conn.execute(
        "INSERT INTO checkpoints (experiment_id, seed, fold, kind, path, metric_value, epoch, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id, seed, fold, kind) DO UPDATE SET "
        "path=excluded.path, metric_value=excluded.metric_value, epoch=excluded.epoch, updated_at=excluded.updated_at",
        (experiment_id, seed, fold, kind, path, metric_value, epoch, _now()),
    )


def get_checkpoint(conn: sqlite3.Connection, experiment_id: int, seed: int, fold: int,
                    kind: str) -> dict | None:
    row = conn.execute(
        "SELECT path, metric_value, epoch FROM checkpoints WHERE experiment_id=? AND seed=? AND fold=? AND kind=?",
        (experiment_id, seed, fold, kind),
    ).fetchone()
    return None if row is None else {"path": row[0], "metric_value": row[1], "epoch": row[2]}


def record_final_result(conn: sqlite3.Connection, experiment_id: int, metrics: dict, wandb_url: str) -> None:
    a, b, c = metrics["auroc"], metrics["balanced_accuracy"], metrics["brier"]
    conn.execute(
        "INSERT INTO final_results (experiment_id, auroc, auroc_ci_lo, auroc_ci_hi, balanced_accuracy, "
        "balanced_accuracy_ci_lo, balanced_accuracy_ci_hi, brier, brier_ci_lo, brier_ci_hi, wandb_url, timestamp) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (experiment_id, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], wandb_url, _now()),
    )
    conn.execute("UPDATE experiments SET status='completed', updated_at=? WHERE id=?", (_now(), experiment_id))
