"""Tests for the adjacent domain watcher agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "snow-town"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.domain_watcher import (
    _make_signal_id,
    _search_hn,
    run_agent,
)


SAMPLE_HN_RESPONSE = {
    "hits": [
        {
            "objectID": "12345",
            "title": "AI Transforms Home Health Nursing",
            "url": "https://example.com/health-ai",
            "points": 250,
            "num_comments": 80,
            "created_at": "2026-02-20T10:00:00.000Z",
            "author": "healthdev",
        },
        {
            "objectID": "67890",
            "title": "New Solo Dev AI Workflow",
            "url": None,
            "points": 50,
            "num_comments": 15,
            "created_at": "2026-02-19T10:00:00.000Z",
            "author": "indiehacker",
        },
    ]
}


class TestMakeSignalId:

    def test_deterministic(self):
        id1 = _make_signal_id("hn-12345")
        id2 = _make_signal_id("hn-12345")
        assert id1 == id2
        assert id1.startswith("domain-")


class TestSearchHN:

    @patch("research_agents.agents.domain_watcher.httpx.get")
    def test_parse_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_HN_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        stories = _search_hn("healthcare AI", max_results=5)
        assert len(stories) == 2
        assert stories[0]["title"] == "AI Transforms Home Health Nursing"
        assert stories[0]["points"] == 250
        # Null URL should get HN fallback
        assert "news.ycombinator.com" in stories[1]["url"]

    @patch("research_agents.agents.domain_watcher.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.HTTPError("Timeout")
        stories = _search_hn("test")
        assert stories == []


class TestRunAgent:

    @patch("research_agents.agents.domain_watcher._search_hn")
    @patch("research_agents.agents.domain_watcher.DOMAIN_WATCH_QUERIES", ["healthcare AI"])
    def test_dry_run_no_writes(self, mock_search, store):
        mock_search.return_value = [
            {
                "objectID": "99999",
                "title": "Test Story",
                "url": "https://example.com",
                "points": 100,
                "num_comments": 20,
                "created_at": "2026-02-20",
                "author": "test",
            }
        ]

        with patch("research_agents.agents.domain_watcher.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result

    @patch("research_agents.agents.domain_watcher.assess_relevance")
    @patch("research_agents.agents.domain_watcher._search_hn")
    @patch("research_agents.agents.domain_watcher.get_client")
    @patch("research_agents.agents.domain_watcher.DOMAIN_WATCH_QUERIES", ["healthcare AI"])
    @patch("research_agents.agents.domain_watcher.time.sleep")
    def test_high_only_filter(self, mock_sleep, mock_client, mock_search, mock_assess, store):
        """Domain watcher should only write HIGH relevance signals."""
        mock_search.return_value = [
            {
                "objectID": "11111",
                "title": "Healthcare AI Breakthrough",
                "url": "https://example.com/1",
                "points": 300,
                "num_comments": 100,
                "created_at": "2026-02-20",
                "author": "dev",
            },
            {
                "objectID": "22222",
                "title": "Medium Relevance Story",
                "url": "https://example.com/2",
                "points": 50,
                "num_comments": 10,
                "created_at": "2026-02-19",
                "author": "dev2",
            },
        ]

        mock_assess.side_effect = [
            {
                "relevance": "high",
                "relevance_rationale": "Direct healthcare AI impact",
                "tags": ["healthcare"],
                "domain": "healthcare-ai",
                "persona_tags": [],
            },
            {
                "relevance": "medium",
                "relevance_rationale": "Tangentially related",
                "tags": [],
                "domain": None,
                "persona_tags": [],
            },
        ]

        with patch("research_agents.agents.domain_watcher.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        assert "1 skipped" in result

        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].source == SignalSource.DOMAIN_WATCH
        assert signals[0].domain == "healthcare-ai"
