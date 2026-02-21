"""Tests for the idea surfacer agent."""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "snow-town"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import ResearchSignal, SignalRelevance, SignalSource

from research_agents.agents.idea_surfacer import (
    _get_recent_signals,
    _write_idea_to_um,
    run_agent,
)
from research_agents.signal_writer import write_signal


@pytest.fixture
def um_db(tmp_path) -> Path:
    """Create a temp caught_ideas.db."""
    db_path = tmp_path / "caught_ideas.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS caught_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            raw_content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            source_context TEXT,
            caught_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            processed_at TIMESTAMP,
            factory_id TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()
    return db_path


class TestWriteIdeaToUm:

    def test_writes_idea(self, um_db: Path):
        idea_id = _write_idea_to_um(
            title="Test Idea",
            description="A test idea from research signals",
            tags=["mcp", "testing"],
            db_path=um_db,
        )
        assert idea_id > 0

        conn = sqlite3.connect(str(um_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM caught_ideas WHERE id = ?", (idea_id,)).fetchone()
        conn.close()

        assert row["title"] == "Test Idea"
        assert row["raw_content"] == "A test idea from research signals"
        assert row["source_context"] == "research-agents:idea-surfacer"
        tags = json.loads(row["tags"])
        assert "machine-surfaced" in tags
        assert "mcp" in tags
        assert row["status"] == "pending"

    def test_creates_table_if_missing(self, tmp_path):
        db_path = tmp_path / "new_ideas.db"
        idea_id = _write_idea_to_um(
            title="New DB Idea",
            description="Tests table creation",
            tags=[],
            db_path=db_path,
        )
        assert idea_id > 0
        assert db_path.exists()


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
    def test_full_run(self, mock_consume, mock_synth, store, um_db):
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
            patch("research_agents.agents.idea_surfacer._get_um_db_path", return_value=um_db),
        ):
            result = run_agent(dry_run=False)

        assert "1 ideas" in result

        # Verify idea was written to UM
        conn = sqlite3.connect(str(um_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM caught_ideas").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["title"] == "MCP Pattern Library"
        assert "machine-surfaced" in json.loads(rows[0]["tags"])

        # Verify signals were marked consumed
        mock_consume.assert_called_once_with(["synth-001"])
