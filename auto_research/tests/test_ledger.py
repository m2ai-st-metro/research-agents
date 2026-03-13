"""Tests for the experiment ledger."""

import sqlite3
from pathlib import Path

import pytest

from auto_research.ledger import (
    get_committed_this_week,
    get_experiment_summary,
    get_last_weekly_baseline,
    get_winners,
    init_db,
    log_experiment,
    mark_committed,
    mark_rolled_back,
    mark_validated,
    save_weekly_baseline,
)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a test database."""
    return init_db(tmp_path / "test_experiments.db")


def test_init_db_creates_tables(db: sqlite3.Connection):
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row[0] for row in tables}
    assert "experiments" in names
    assert "weekly_baselines" in names


def test_log_experiment(db: sqlite3.Connection):
    exp_id = log_experiment(
        conn=db,
        agent="arxiv",
        param_name="ARXIV_SEARCH_QUERIES[0]",
        baseline_value="MCP model context protocol",
        variant_value="model context protocol agent tools",
        baseline_signals=15,
        variant_signals=12,
        baseline_ndr=0.10,
        variant_ndr=0.25,
        baseline_avg_score=4.8,
        variant_avg_score=5.2,
        improvement_pct=1.5,
        status="completed",
    )
    assert exp_id == 1

    row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    assert row["agent"] == "arxiv"
    assert row["baseline_ndr"] == 0.10
    assert row["variant_ndr"] == 0.25
    assert row["improvement_pct"] == 1.5


def test_mark_validated(db: sqlite3.Connection):
    exp_id = log_experiment(
        conn=db,
        agent="arxiv",
        param_name="test",
        baseline_value="q1",
        variant_value="q2",
        baseline_signals=10,
        variant_signals=10,
        baseline_ndr=0.1,
        variant_ndr=0.3,
        baseline_avg_score=5.0,
        variant_avg_score=5.5,
        improvement_pct=2.0,
    )

    mark_validated(db, exp_id, claude_ndr=0.35)
    row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    assert row["claude_validated"] == 1
    assert row["claude_ndr"] == 0.35


def test_mark_committed(db: sqlite3.Connection):
    exp_id = log_experiment(
        conn=db, agent="arxiv", param_name="test",
        baseline_value="q1", variant_value="q2",
        baseline_signals=10, variant_signals=10,
        baseline_ndr=0.1, variant_ndr=0.3,
        baseline_avg_score=5.0, variant_avg_score=5.5,
        improvement_pct=2.0,
    )

    mark_committed(db, exp_id, "abc123def")
    row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    assert row["committed"] == 1
    assert row["commit_sha"] == "abc123def"


def test_get_winners(db: sqlite3.Connection):
    # Below threshold
    log_experiment(
        conn=db, agent="arxiv", param_name="test",
        baseline_value="q1", variant_value="q2",
        baseline_signals=10, variant_signals=10,
        baseline_ndr=0.1, variant_ndr=0.11,
        baseline_avg_score=5.0, variant_avg_score=5.0,
        improvement_pct=0.10,
    )

    # Above threshold
    log_experiment(
        conn=db, agent="tool_monitor", param_name="test",
        baseline_value="q3", variant_value="q4",
        baseline_signals=10, variant_signals=10,
        baseline_ndr=0.1, variant_ndr=0.3,
        baseline_avg_score=5.0, variant_avg_score=5.5,
        improvement_pct=2.0,
    )

    winners = get_winners(db, threshold=0.15)
    assert len(winners) == 1
    assert winners[0]["agent"] == "tool_monitor"


def test_get_experiment_summary(db: sqlite3.Connection):
    log_experiment(
        conn=db, agent="arxiv", param_name="test",
        baseline_value="q1", variant_value="q2",
        baseline_signals=10, variant_signals=10,
        baseline_ndr=0.1, variant_ndr=0.3,
        baseline_avg_score=5.0, variant_avg_score=5.5,
        improvement_pct=2.0,
    )

    summary = get_experiment_summary(db)
    assert summary["total_experiments"] == 1
    assert summary["winners"] == 1
    assert summary["committed"] == 0


def test_weekly_baseline(db: sqlite3.Connection):
    save_weekly_baseline(db, 0.13, 5.2, {"arxiv": {"ndr": 0.15}})
    baseline = get_last_weekly_baseline(db)
    assert baseline is not None
    assert baseline["overall_ndr"] == 0.13
    assert baseline["overall_avg_score"] == 5.2


def test_mark_rolled_back(db: sqlite3.Connection):
    ids = []
    for i in range(3):
        exp_id = log_experiment(
            conn=db, agent="arxiv", param_name=f"test{i}",
            baseline_value=f"q{i}", variant_value=f"v{i}",
            baseline_signals=10, variant_signals=10,
            baseline_ndr=0.1, variant_ndr=0.3,
            baseline_avg_score=5.0, variant_avg_score=5.5,
            improvement_pct=2.0,
        )
        mark_committed(db, exp_id, f"sha{i}")
        ids.append(exp_id)

    mark_rolled_back(db, ids)

    for exp_id in ids:
        row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        assert row["rolled_back"] == 1
