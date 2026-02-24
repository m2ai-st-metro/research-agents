"""Tests for the idea surfacer agent."""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "st-factory"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import ResearchSignal, SignalRelevance, SignalSource

from research_agents.agents.idea_surfacer import (
    _get_recent_signals,
    run_agent,
)
from research_agents.agents.ideaforge_writer import write_idea_to_ideaforge
from research_agents.signal_writer import write_signal


# --- IdeaForge schema for test fixtures ---

IDEAFORGE_IDEAS_SCHEMA = """
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
    ultra_magnus_id INTEGER
);
"""


@pytest.fixture
def ideaforge_db(tmp_path) -> Path:
    """Create a temp IdeaForge ideas DB."""
    db_path = tmp_path / "ideaforge.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(IDEAFORGE_IDEAS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


class TestWriteIdeaToIdeaForge:

    def test_writes_idea_with_correct_schema(self, ideaforge_db: Path):
        idea_id = write_idea_to_ideaforge(
            title="Test Idea",
            description="A test idea from research signals",
            tags=["mcp", "testing"],
            source_signal_ids=["sig-001", "sig-002"],
            db_path=ideaforge_db,
        )
        assert idea_id > 0

        conn = sqlite3.connect(str(ideaforge_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        conn.close()

        assert row["title"] == "Test Idea"
        assert row["description"] == "A test idea from research signals"
        assert row["status"] == "unscored"
        assert row["signal_count"] == 2
        assert row["problem_statement"] == ""
        assert row["target_audience"] == ""
        assert row["synthesized_at"] is not None

        # source_signals should be the signal IDs
        source_signals = json.loads(row["source_signals"])
        assert source_signals == ["sig-001", "sig-002"]

        # source_subreddits holds provenance + tags
        provenance = json.loads(row["source_subreddits"])
        assert "research-agents:idea-surfacer" in provenance
        assert "mcp" in provenance
        assert "testing" in provenance

    def test_creates_table_if_missing(self, tmp_path):
        db_path = tmp_path / "new_ideaforge.db"
        idea_id = write_idea_to_ideaforge(
            title="New DB Idea",
            description="Tests table creation",
            tags=[],
            source_signal_ids=[],
            db_path=db_path,
        )
        assert idea_id > 0
        assert db_path.exists()

        # Verify status is unscored
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        conn.close()
        assert row["status"] == "unscored"

    def test_empty_tags_and_signals(self, ideaforge_db: Path):
        idea_id = write_idea_to_ideaforge(
            title="Minimal Idea",
            description="No tags or signals",
            tags=[],
            source_signal_ids=[],
            db_path=ideaforge_db,
        )
        assert idea_id > 0

        conn = sqlite3.connect(str(ideaforge_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        conn.close()

        assert row["signal_count"] == 0
        assert json.loads(row["source_signals"]) == []
        # provenance is still present even with no tags
        provenance = json.loads(row["source_subreddits"])
        assert "research-agents:idea-surfacer" in provenance

    def test_multiple_ideas_get_unique_ids(self, ideaforge_db: Path):
        id1 = write_idea_to_ideaforge(
            title="Idea 1", description="First", tags=[], source_signal_ids=[], db_path=ideaforge_db,
        )
        id2 = write_idea_to_ideaforge(
            title="Idea 2", description="Second", tags=[], source_signal_ids=[], db_path=ideaforge_db,
        )
        assert id1 != id2
        assert id1 > 0
        assert id2 > 0


class TestGetRecentSignals:

    def test_returns_recent_unconsumed(self, store):
        write_signal(
            signal_id="recent-high",
            source=SignalSource.ARXIV_HF,
            title="Recent High",
            summary="Test",
            relevance=SignalRelevance.HIGH,
            store=store,
        )
        write_signal(
            signal_id="recent-low",
            source=SignalSource.ARXIV_HF,
            title="Recent Low",
            summary="Test",
            relevance=SignalRelevance.LOW,
            store=store,
        )

        with patch("research_agents.agents.idea_surfacer.get_store", return_value=store):
            signals = _get_recent_signals(days=7)

        # Should include high but not low
        assert len(signals) == 1
        assert signals[0].signal_id == "recent-high"


class TestRunAgent:

    def test_dry_run(self, store):
        write_signal(
            signal_id="dry-001",
            source=SignalSource.ARXIV_HF,
            title="Dry Run Signal",
            summary="Test",
            relevance=SignalRelevance.HIGH,
            store=store,
        )

        with patch("research_agents.agents.idea_surfacer.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "DRY RUN" in result
        assert "1 signals" in result

    def test_no_signals(self, store):
        with patch("research_agents.agents.idea_surfacer.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "No recent unconsumed signals" in result

    @patch("research_agents.agents.idea_surfacer._synthesize_ideas")
    @patch("research_agents.agents.idea_surfacer._mark_signals_consumed")
    def test_full_run_writes_to_ideaforge(self, mock_consume, mock_synth, store, ideaforge_db):
        write_signal(
            signal_id="synth-001",
            source=SignalSource.ARXIV_HF,
            title="MCP Paper",
            summary="New MCP patterns",
            relevance=SignalRelevance.HIGH,
            store=store,
        )

        mock_synth.return_value = [
            {
                "title": "MCP Pattern Library",
                "description": "Build a reusable pattern library for MCP servers based on recent research",
                "tags": ["mcp", "patterns"],
                "source_signal_ids": ["synth-001"],
            }
        ]

        with (
            patch("research_agents.agents.idea_surfacer.get_store", return_value=store),
            patch(
                "research_agents.agents.idea_surfacer.write_idea_to_ideaforge",
                wraps=lambda **kw: write_idea_to_ideaforge(**kw, db_path=ideaforge_db),
            ) as mock_write,
        ):
            result = run_agent(dry_run=False)

        assert "1 ideas" in result

        # Verify idea was written to IdeaForge (not UM)
        conn = sqlite3.connect(str(ideaforge_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ideas").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["title"] == "MCP Pattern Library"
        assert rows[0]["status"] == "unscored"
        assert rows[0]["signal_count"] == 1
        source_signals = json.loads(rows[0]["source_signals"])
        assert "synth-001" in source_signals

        # Verify provenance
        provenance = json.loads(rows[0]["source_subreddits"])
        assert "research-agents:idea-surfacer" in provenance

        # Verify signals were marked consumed
        mock_consume.assert_called_once_with(["synth-001"])
