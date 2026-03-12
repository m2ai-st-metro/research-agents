"""Tests for the manual signal writer utility."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Snow-Town contracts are importable
SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "st-factory"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.manual_signal_writer import (
    _is_url,
    _make_signal_id,
    ingest_signal,
)


class TestMakeSignalId:

    def test_format(self):
        sid = _make_signal_id("https://example.com")
        assert sid.startswith("manual-")
        assert len(sid) == 7 + 12

    def test_deterministic(self):
        assert _make_signal_id("foo") == _make_signal_id("foo")

    def test_different_inputs_different_ids(self):
        assert _make_signal_id("a") != _make_signal_id("b")


class TestIsUrl:

    def test_https_is_url(self):
        assert _is_url("https://example.com") is True

    def test_http_is_url(self):
        assert _is_url("http://example.com") is True

    def test_topic_not_url(self):
        assert _is_url("interesting idea about LLM self-play") is False

    def test_empty_string(self):
        assert _is_url("") is False


class TestIngestSignal:

    def test_dry_run(self):
        with patch("research_agents.agents.manual_signal_writer.signal_exists", return_value=False):
            result = ingest_signal("https://example.com/article", dry_run=True)
        assert "DRY RUN" in result

    @patch("research_agents.agents.manual_signal_writer.signal_exists", return_value=True)
    def test_dedup_returns_early(self, mock_exists):
        result = ingest_signal("https://example.com/already-seen")
        assert "Already in signal store" in result

    @patch("research_agents.agents.manual_signal_writer.write_signal")
    @patch("research_agents.agents.manual_signal_writer.get_client")
    @patch("research_agents.agents.manual_signal_writer.assess_relevance")
    @patch("research_agents.agents.manual_signal_writer._fetch_url_content")
    @patch("research_agents.agents.manual_signal_writer.signal_exists", return_value=False)
    def test_url_ingest_writes_signal(
        self, mock_exists, mock_fetch, mock_assess, mock_client, mock_write
    ):
        mock_fetch.return_value = ("Article Title", "Article summary text")
        mock_assess.return_value = {
            "relevance": "high",
            "relevance_rationale": "Very relevant",
            "tags": ["mcp"],
            "domain": "ai-agents",
            "persona_tags": [],
        }

        result = ingest_signal("https://example.com/article")
        assert mock_write.called
        assert "Signal saved" in result
        assert "high" in result

        # Verify write_signal was called with correct source
        call_kwargs = mock_write.call_args
        assert call_kwargs.kwargs["source"] == SignalSource.MANUAL

    @patch("research_agents.agents.manual_signal_writer.write_signal")
    @patch("research_agents.agents.manual_signal_writer.get_client")
    @patch("research_agents.agents.manual_signal_writer.assess_relevance")
    @patch("research_agents.agents.manual_signal_writer.signal_exists", return_value=False)
    def test_topic_ingest_no_url_fetch(
        self, mock_exists, mock_assess, mock_client, mock_write
    ):
        mock_assess.return_value = {
            "relevance": "medium",
            "relevance_rationale": "Somewhat relevant",
            "tags": ["llm"],
            "domain": None,
            "persona_tags": ["hopper"],
        }

        result = ingest_signal("LLM self-play for code review")
        assert mock_write.called
        assert "Signal saved" in result

        # URL should be None for topic-only input
        call_kwargs = mock_write.call_args
        assert call_kwargs.kwargs["url"] is None
        # Should have "manual" tag
        tags = call_kwargs.kwargs["tags"]
        assert "manual" in tags
        assert "persona:hopper" in tags
