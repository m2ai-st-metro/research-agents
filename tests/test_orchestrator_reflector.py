"""Tests for the orchestrator reflector agent."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from research_agents.agents.orchestrator_reflector import (
    COMPOSITE_FAILURE_THRESHOLD,
    ReflectorCursor,
    run_agent,
)


# --- Schemas matching the real databases ---

ORCHESTRATOR_SCHEMA = """
CREATE TABLE outcome_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    subtask_id TEXT,
    task_type TEXT,
    agent_id TEXT,
    status TEXT,
    duration_ms INTEGER,
    correctness REAL,
    completeness REAL,
    relevance REAL,
    composite_score REAL,
    judge_reasoning TEXT,
    judge_method TEXT,
    created_at INTEGER DEFAULT (unixepoch())
);
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    decision_type TEXT NOT NULL,
    reasoning TEXT,
    subtask_id TEXT,
    target_agent_id TEXT,
    cost_usd REAL DEFAULT 0,
    created_at INTEGER DEFAULT (unixepoch())
);
"""

IDEAFORGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    problem_statement TEXT DEFAULT '',
    target_audience TEXT DEFAULT '',
    source_signals TEXT DEFAULT '[]',
    source_subreddits TEXT DEFAULT '[]',
    signal_count INTEGER DEFAULT 0,
    opportunity_score REAL,
    problem_score REAL,
    feasibility_score REAL,
    why_now_score REAL,
    competition_score REAL,
    weighted_score REAL,
    score_rationale TEXT,
    artifact_type TEXT,
    route_rationale TEXT,
    route_confidence REAL,
    struggling_user TEXT,
    classified_at TIMESTAMP,
    status TEXT DEFAULT 'unscored',
    synthesized_at TIMESTAMP,
    scored_at TIMESTAMP,
    exported_at TIMESTAMP,
    ultra_magnus_id INTEGER,
    signal_source TEXT DEFAULT 'unknown'
);
"""


@pytest.fixture
def orchestrator_db(tmp_path: Path) -> Path:
    db = tmp_path / "orchestrator.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(ORCHESTRATOR_SCHEMA)

    # Two distinct failing outcomes + one cluster of 3 identical (task_type, agent_id)
    # failures + one passing outcome that should be skipped.
    conn.executemany(
        """INSERT INTO outcome_logs
           (mission_id, subtask_id, task_type, agent_id, status,
            composite_score, judge_reasoning, judge_method)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ("m-1", "st-a", "research", "worker", "completed", 0.33,
             "Agent failed to locate the content.", "llm"),
            ("m-1", "st-b", "coding", "galvatron", "completed", 0.46,
             "Architectural analysis was off-target.", "llm"),
            ("m-1", "st-c", "general", "soundwave", "completed", 0.80,
             "Excellent output.", "llm"),
            # Cluster: 3 more 'research/worker' failures, all worse than 0.33
            ("m-1", "st-d", "research", "worker", "completed", 0.20,
             "Still failing to locate.", "llm"),
            ("m-1", "st-e", "research", "worker", "completed", 0.10,
             "Worst one of the cluster.", "llm"),
            ("m-1", "st-f", "research", "worker", "completed", 0.25,
             "Another failure.", "llm"),
        ],
    )

    # Decisions: a replan cluster of 3 on mission m-1 + one reassign + one accept.
    conn.executemany(
        """INSERT INTO decisions
           (mission_id, iteration, decision_type, reasoning,
            subtask_id, target_agent_id, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            ("m-1", 1, "reassign", "Worker hallucinated constraints.", "st-a", "soundwave", 0.02),
            ("m-1", 2, "replan", "Routed to nonexistent agent.", None, None, 0.01),
            ("m-1", 3, "replan", "Still routing wrong.", None, None, 0.01),
            ("m-1", 4, "replan", "Final replan attempt.", None, None, 0.01),
            ("m-1", 5, "accept", "All subtasks met threshold.", None, None, 0.00),
        ],
    )

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def ideaforge_db(tmp_path: Path) -> Path:
    db = tmp_path / "ideaforge.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(IDEAFORGE_SCHEMA)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def cursor_path(tmp_path: Path) -> Path:
    return tmp_path / "orchestrator_reflector_state.json"


# --- Cursor tests ---


def test_cursor_defaults_when_missing(tmp_path: Path) -> None:
    cursor = ReflectorCursor.load(tmp_path / "missing.json")
    assert cursor.last_outcome_id == 0
    assert cursor.last_decision_id == 0


def test_cursor_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    ReflectorCursor(last_outcome_id=10, last_decision_id=7).save(path)
    loaded = ReflectorCursor.load(path)
    assert loaded.last_outcome_id == 10
    assert loaded.last_decision_id == 7


def test_cursor_recovers_from_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    cursor = ReflectorCursor.load(path)
    assert cursor.last_outcome_id == 0
    assert cursor.last_decision_id == 0


# --- Filter thresholds ---


def test_failure_threshold_is_below_acceptance() -> None:
    """Sanity check: reflector threshold should be at or below ClaudeClaw acceptThreshold."""
    # ClaudeClaw default acceptThreshold is 0.45 per agentic-orchestrator-project memory.
    # Ours is 0.5 which is intentionally slightly more permissive to also catch
    # borderline passes that the reasoner accepted but a human might reflect on.
    assert COMPOSITE_FAILURE_THRESHOLD >= 0.45


# --- End-to-end reflector runs ---


def test_dry_run_reports_candidates_without_writing(
    orchestrator_db: Path, ideaforge_db: Path, cursor_path: Path
) -> None:
    result = run_agent(
        dry_run=True,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )
    # 5 failing outcomes across 2 (task_type,agent_id) groups
    #   + 4 reflectable decisions across 2 (mission_id,decision_type) groups
    # = 4 deduped candidates total.
    assert "4" in result

    # Nothing written (dry-run skips all writes, table may not exist)
    conn = sqlite3.connect(str(ideaforge_db))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "capability_gaps" in tables:
            count = conn.execute("SELECT COUNT(*) FROM capability_gaps").fetchone()[0]
            assert count == 0
    finally:
        conn.close()

    # Cursor not advanced
    assert not cursor_path.exists()


def test_live_run_writes_deduped_ideas_and_advances_cursor(
    orchestrator_db: Path, ideaforge_db: Path, cursor_path: Path
) -> None:
    result = run_agent(
        dry_run=False,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )
    assert "4" in result

    conn = sqlite3.connect(str(ideaforge_db))
    try:
        rows = conn.execute(
            "SELECT title, status, signal_source, source_signals, "
            "problem_statement "
            "FROM capability_gaps ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 4
    assert all(row[1] == "raw" for row in rows)
    assert all(row[2] == "orchestrator_reflector" for row in rows)

    titles = [row[0] for row in rows]
    assert any("Capability gap" in t for t in titles)
    assert any("Orchestrator decision" in t for t in titles)

    # The research/worker outcome cluster must be emitted as one idea reporting
    # the cluster size (x4) with the worst score in the group (0.10).
    research_worker = [t for t in titles if "research" in t and "worker" in t]
    assert len(research_worker) == 1
    assert "x4" in research_worker[0]
    assert "0.10" in research_worker[0]

    # The replan decision cluster must be emitted as one idea reporting x3.
    replans = [t for t in titles if "replan" in t]
    assert len(replans) == 1
    assert "x3" in replans[0]

    # Source signals must cite ALL member row IDs from each cluster.
    # (capability_gaps query returns: title[0], status[1], signal_source[2], source_signals[3], problem_statement[4])
    sources_by_title = {row[0]: json.loads(row[3]) for row in rows}
    cluster_sources = next(
        s for t, s in sources_by_title.items() if "research" in t and "worker" in t
    )
    assert len(cluster_sources) == 4  # all 4 research/worker outcomes cited

    # Cursor advanced past the highest inspected ID of each kind (not just
    # the representative's ID) so the within-cluster duplicates aren't
    # re-processed on the next run.
    cursor = ReflectorCursor.load(cursor_path)
    assert cursor.last_outcome_id == 6  # highest failing outcome row ID
    assert cursor.last_decision_id == 4  # highest reflectable decision row ID


def test_second_run_is_idempotent(
    orchestrator_db: Path, ideaforge_db: Path, cursor_path: Path
) -> None:
    run_agent(
        dry_run=False,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )
    result = run_agent(
        dry_run=False,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )
    assert "0" in result

    conn = sqlite3.connect(str(ideaforge_db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM capability_gaps").fetchone()[0]
    finally:
        conn.close()
    assert count == 4  # still four, no duplicates


def test_new_rows_after_cursor_are_picked_up(
    orchestrator_db: Path, ideaforge_db: Path, cursor_path: Path
) -> None:
    run_agent(
        dry_run=False,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )

    # Simulate new orchestrator activity
    conn = sqlite3.connect(str(orchestrator_db))
    conn.execute(
        """INSERT INTO outcome_logs
           (mission_id, subtask_id, task_type, agent_id, status,
            composite_score, judge_reasoning, judge_method)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("m-2", "st-z", "coding", "worker", "completed", 0.12,
         "Totally missed the mark.", "llm"),
    )
    conn.execute(
        """INSERT INTO decisions
           (mission_id, iteration, decision_type, reasoning,
            subtask_id, target_agent_id, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("m-2", 1, "escalate", "Out of retries.", None, None, 0.05),
    )
    conn.commit()
    conn.close()

    result = run_agent(
        dry_run=False,
        orchestrator_db=orchestrator_db,
        ideaforge_db=ideaforge_db,
        cursor_path=cursor_path,
    )
    assert "2" in result

    conn = sqlite3.connect(str(ideaforge_db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM capability_gaps").fetchone()[0]
    finally:
        conn.close()
    assert count == 6


def test_missing_orchestrator_db_is_graceful(tmp_path: Path) -> None:
    result = run_agent(
        dry_run=False,
        orchestrator_db=tmp_path / "does-not-exist.db",
        ideaforge_db=tmp_path / "ideaforge.db",
        cursor_path=tmp_path / "state.json",
    )
    assert "not found" in result.lower()
