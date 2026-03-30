"""Tests for the trend analyzer agent."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Snow-Town contracts are importable
ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from contracts.research_signal import ResearchSignal, SignalRelevance, SignalSource

from research_agents.agents.trend_analyzer import (
    _cluster_signals,
    _detect_rising_themes,
    run_agent,
)


def _make_signal(
    signal_id: str,
    tags: list[str],
    domain: str | None,
    days_ago: int = 0,
) -> ResearchSignal:
    """Helper to create a ResearchSignal with controlled emitted_at."""
    return ResearchSignal(
        signal_id=signal_id,
        source=SignalSource.ARXIV_HF,
        title=f"Signal {signal_id}",
        summary="Test summary for signal",
        relevance=SignalRelevance.HIGH,
        tags=tags,
        domain=domain,
        emitted_at=datetime.now() - timedelta(days=days_ago),
    )


class TestClusterSignals:

    def test_groups_by_domain(self):
        signals = [
            _make_signal("s1", ["mcp"], "ai-agents"),
            _make_signal("s2", ["mcp", "llm"], "ai-agents"),
            _make_signal("s3", ["hipaa"], "healthcare-ai"),
        ]
        clusters = _cluster_signals(signals)
        by_domain = clusters["by_domain"]
        assert "ai-agents" in by_domain
        assert len(by_domain["ai-agents"]) == 2  # type: ignore[arg-type]
        assert clusters["tag_counts"]["mcp"] == 2  # type: ignore[index]

    def test_excludes_persona_tags(self):
        signals = [_make_signal("s1", ["mcp", "persona:carmack"], "ai-agents")]
        clusters = _cluster_signals(signals)
        assert "persona:carmack" not in clusters["tag_counts"]  # type: ignore[operator]
        assert "mcp" in clusters["tag_counts"]  # type: ignore[operator]

    def test_null_domain_becomes_general(self):
        signals = [_make_signal("s1", ["llm"], None)]
        clusters = _cluster_signals(signals)
        assert "general" in clusters["by_domain"]  # type: ignore[operator]

    def test_source_counting(self):
        signals = [
            _make_signal("s1", [], "ai-agents"),
            _make_signal("s2", [], "ai-agents"),
        ]
        clusters = _cluster_signals(signals)
        assert clusters["by_source"]["arxiv_hf"] == 2  # type: ignore[index]


class TestDetectRisingThemes:

    def test_rising_tag_detected(self):
        # Tag "mcp" appears only in second half (recent)
        signals = [
            _make_signal("s1", ["mcp"], "ai-agents", days_ago=2),
            _make_signal("s2", ["mcp"], "ai-agents", days_ago=1),
            _make_signal("s3", ["mcp"], "ai-agents", days_ago=0),
        ]
        trends = _detect_rising_themes(signals, window_days=14)
        rising_tags = [t for t, _, _ in trends["rising"]]
        assert "mcp" in rising_tags

    def test_falling_tag_detected(self):
        # Tag "old-topic" only in first half (older)
        signals = [
            _make_signal("s1", ["old-topic"], "ai-agents", days_ago=13),
            _make_signal("s2", ["old-topic"], "ai-agents", days_ago=12),
            _make_signal("s3", ["old-topic"], "ai-agents", days_ago=11),
        ]
        trends = _detect_rising_themes(signals, window_days=14)
        falling_tags = [t for t, _, _ in trends["falling"]]
        assert "old-topic" in falling_tags

    def test_excludes_persona_tags(self):
        signals = [
            _make_signal("s1", ["persona:hopper"], "ai-agents", days_ago=1),
            _make_signal("s2", ["persona:hopper"], "ai-agents", days_ago=0),
        ]
        trends = _detect_rising_themes(signals, window_days=14)
        all_rising = [t for t, _, _ in trends["rising"]]
        assert "persona:hopper" not in all_rising


class TestRunAgent:

    @patch("research_agents.agents.trend_analyzer._load_signals")
    def test_insufficient_signals_skips(self, mock_load):
        mock_load.return_value = [_make_signal("s1", [], None)]
        result = run_agent(dry_run=False)
        assert "Insufficient" in result

    @patch("research_agents.agents.trend_analyzer._load_signals")
    def test_dry_run_no_api_calls(self, mock_load):
        mock_load.return_value = [
            _make_signal(f"s{i}", ["mcp"], "ai-agents") for i in range(10)
        ]
        result = run_agent(dry_run=True)
        assert "DRY RUN" in result
        assert "10 signals" in result
