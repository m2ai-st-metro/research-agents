"""Experiment ledger — SQLite storage for experiment history and results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from .config import EXPERIMENTS_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent TEXT NOT NULL,
    param_name TEXT NOT NULL,
    baseline_value TEXT NOT NULL,
    variant_value TEXT NOT NULL,
    baseline_signals INTEGER NOT NULL DEFAULT 0,
    variant_signals INTEGER NOT NULL DEFAULT 0,
    baseline_ndr REAL,
    variant_ndr REAL,
    baseline_avg_score REAL,
    variant_avg_score REAL,
    improvement_pct REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    claude_validated INTEGER DEFAULT 0,
    claude_ndr REAL,
    committed INTEGER DEFAULT 0,
    commit_sha TEXT,
    rolled_back INTEGER DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS weekly_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start TEXT NOT NULL,
    week_end TEXT NOT NULL,
    overall_ndr REAL NOT NULL,
    overall_avg_score REAL NOT NULL,
    agent_metrics TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp);
CREATE INDEX IF NOT EXISTS idx_experiments_agent ON experiments(agent);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_weekly_week_start ON weekly_baselines(week_start);
"""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize the experiment ledger database."""
    path = db_path or EXPERIMENTS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_experiment(
    conn: sqlite3.Connection,
    agent: str,
    param_name: str,
    baseline_value: str,
    variant_value: str,
    baseline_signals: int,
    variant_signals: int,
    baseline_ndr: float | None,
    variant_ndr: float | None,
    baseline_avg_score: float | None,
    variant_avg_score: float | None,
    improvement_pct: float | None,
    status: str = "completed",
    notes: str | None = None,
) -> int:
    """Log an experiment result. Returns the experiment ID."""
    cursor = conn.execute(
        """INSERT INTO experiments (
            timestamp, agent, param_name, baseline_value, variant_value,
            baseline_signals, variant_signals, baseline_ndr, variant_ndr,
            baseline_avg_score, variant_avg_score, improvement_pct, status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            agent,
            param_name,
            baseline_value,
            variant_value,
            baseline_signals,
            variant_signals,
            baseline_ndr,
            variant_ndr,
            baseline_avg_score,
            variant_avg_score,
            improvement_pct,
            status,
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def mark_validated(
    conn: sqlite3.Connection,
    experiment_id: int,
    claude_ndr: float,
) -> None:
    """Mark an experiment as validated by Claude API."""
    conn.execute(
        "UPDATE experiments SET claude_validated = 1, claude_ndr = ? WHERE id = ?",
        (claude_ndr, experiment_id),
    )
    conn.commit()


def mark_committed(
    conn: sqlite3.Connection,
    experiment_id: int,
    commit_sha: str,
) -> None:
    """Mark an experiment's mutation as committed to config."""
    conn.execute(
        "UPDATE experiments SET committed = 1, commit_sha = ? WHERE id = ?",
        (commit_sha, experiment_id),
    )
    conn.commit()


def mark_rolled_back(
    conn: sqlite3.Connection,
    experiment_ids: list[int],
) -> None:
    """Mark experiments as rolled back."""
    placeholders = ",".join("?" * len(experiment_ids))
    conn.execute(
        f"UPDATE experiments SET rolled_back = 1 WHERE id IN ({placeholders})",
        experiment_ids,
    )
    conn.commit()


def get_winners(
    conn: sqlite3.Connection,
    threshold: float = 0.15,
    limit: int = 10,
) -> list[sqlite3.Row]:
    """Get experiments that beat the improvement threshold."""
    return conn.execute(
        """SELECT * FROM experiments
        WHERE status = 'completed'
          AND improvement_pct >= ?
          AND committed = 0
          AND rolled_back = 0
        ORDER BY improvement_pct DESC
        LIMIT ?""",
        (threshold, limit),
    ).fetchall()


def get_committed_this_week(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Get experiments committed in the current week (for rollback checks)."""
    week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return conn.execute(
        """SELECT * FROM experiments
        WHERE committed = 1
          AND rolled_back = 0
          AND timestamp >= ?
        ORDER BY timestamp""",
        (week_start.isoformat(),),
    ).fetchall()


def save_weekly_baseline(
    conn: sqlite3.Connection,
    overall_ndr: float,
    overall_avg_score: float,
    agent_metrics: dict[str, dict],
) -> int:
    """Save a weekly baseline snapshot."""
    now = datetime.now()
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_end = week_start + timedelta(days=7)

    cursor = conn.execute(
        """INSERT INTO weekly_baselines (
            week_start, week_end, overall_ndr, overall_avg_score,
            agent_metrics, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            week_start.isoformat(),
            week_end.isoformat(),
            overall_ndr,
            overall_avg_score,
            json.dumps(agent_metrics),
            now.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def get_last_weekly_baseline(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Get the most recent weekly baseline."""
    return conn.execute(
        "SELECT * FROM weekly_baselines ORDER BY week_start DESC LIMIT 1"
    ).fetchone()


def get_experiment_summary(conn: sqlite3.Connection) -> dict:
    """Get a summary of all experiments."""
    total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    winners = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE improvement_pct >= 0.15"
    ).fetchone()[0]
    committed = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE committed = 1"
    ).fetchone()[0]
    rolled_back = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE rolled_back = 1"
    ).fetchone()[0]
    validated = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE claude_validated = 1"
    ).fetchone()[0]

    return {
        "total_experiments": total,
        "winners": winners,
        "committed": committed,
        "rolled_back": rolled_back,
        "claude_validated": validated,
    }
