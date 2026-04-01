"""Tests for the GitHub trending harvester agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.github_trending import (
    _make_signal_id,
    _search_trending_repos,
    run_agent,
    GITHUB_TRENDING_LANGUAGES,
)


SAMPLE_GITHUB_RESPONSE = {
    "items": [
        {
            "full_name": "trending/ai-framework",
            "description": "A new AI agent framework for autonomous coding",
            "html_url": "https://github.com/trending/ai-framework",
            "stargazers_count": 200,
            "language": "Python",
            "pushed_at": "2026-03-20T10:00:00Z",
            "created_at": "2026-03-15T10:00:00Z",
            "topics": ["ai", "agents", "framework"],
        },
        {
            "full_name": "trending/cooking-app",
            "description": "Recipe management tool",
            "html_url": "https://github.com/trending/cooking-app",
            "stargazers_count": 100,
            "language": "JavaScript",
            "pushed_at": "2026-03-19T10:00:00Z",
            "created_at": "2026-03-10T10:00:00Z",
            "topics": ["cooking"],
        },
    ]
}


class TestMakeSignalId:

    def test_prefix(self):
        sid = _make_signal_id("trending/ai-framework")
        assert sid.startswith("github_trending-")

    def test_deterministic(self):
        id1 = _make_signal_id("trending/ai-framework")
        id2 = _make_signal_id("trending/ai-framework")
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _make_signal_id("repo-a")
        id2 = _make_signal_id("repo-b")
        assert id1 != id2


class TestSearchTrending:

    @patch("research_agents.agents.github_trending.httpx.get")
    def test_parse_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_GITHUB_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        repos = _search_trending_repos("python", max_results=5)
        assert len(repos) == 2
        assert repos[0]["full_name"] == "trending/ai-framework"
        assert repos[0]["stars"] == 200
        assert "created_at" in repos[0]

    @patch("research_agents.agents.github_trending.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.HTTPError("Rate limited")
        repos = _search_trending_repos("python")
        assert repos == []


class TestRunAgent:

    @patch("research_agents.agents.github_trending._search_trending_repos")
    @patch("research_agents.agents.github_trending.GITHUB_TRENDING_LANGUAGES", ["python"])
    def test_dry_run_no_writes(self, mock_search, store):
        mock_search.return_value = [
            {
                "full_name": "test/trending-repo",
                "description": "Test trending repo",
                "url": "https://github.com/test/trending-repo",
                "stars": 100,
                "language": "Python",
                "pushed_at": "2026-03-20",
                "created_at": "2026-03-15",
                "topics": [],
            }
        ]

        with patch("research_agents.agents.github_trending.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result

    @patch("research_agents.agents.github_trending.assess_relevance")
    @patch("research_agents.agents.github_trending._search_trending_repos")
    @patch("research_agents.agents.github_trending.get_client")
    @patch("research_agents.agents.github_trending.GITHUB_TRENDING_LANGUAGES", ["python"])
    @patch("research_agents.agents.github_trending.time.sleep")
    def test_full_run_writes_signals(self, mock_sleep, mock_client, mock_search, mock_assess, store):
        mock_search.return_value = [
            {
                "full_name": "trending/mcp-toolkit",
                "description": "MCP server toolkit for building tools",
                "url": "https://github.com/trending/mcp-toolkit",
                "stars": 300,
                "language": "Python",
                "pushed_at": "2026-03-20",
                "created_at": "2026-03-15",
                "topics": ["mcp"],
            },
        ]
        mock_assess.return_value = {
            "relevance": "high",
            "relevance_rationale": "Core MCP tooling",
            "tags": ["mcp"],
            "domain": "developer-tools",
        }

        with patch("research_agents.agents.github_trending.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].source == SignalSource.GITHUB_TRENDING

    @patch("research_agents.agents.github_trending.assess_relevance")
    @patch("research_agents.agents.github_trending._search_trending_repos")
    @patch("research_agents.agents.github_trending.get_client")
    @patch("research_agents.agents.github_trending.GITHUB_TRENDING_LANGUAGES", ["python"])
    @patch("research_agents.agents.github_trending.time.sleep")
    def test_low_relevance_skipped(self, mock_sleep, mock_client, mock_search, mock_assess, store):
        mock_search.return_value = [
            {
                "full_name": "trending/irrelevant-repo",
                "description": "Not relevant to ST Metro",
                "url": "https://github.com/trending/irrelevant-repo",
                "stars": 100,
                "language": "Python",
                "pushed_at": "2026-03-20",
                "created_at": "2026-03-15",
                "topics": [],
            },
        ]
        mock_assess.return_value = {
            "relevance": "low",
            "relevance_rationale": "Not relevant",
            "tags": [],
            "domain": None,
        }

        with patch("research_agents.agents.github_trending.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "0 written" in result
        assert "1 skipped (low relevance)" in result

    @patch("research_agents.agents.github_trending.signal_exists")
    @patch("research_agents.agents.github_trending._search_trending_repos")
    @patch("research_agents.agents.github_trending.get_client")
    @patch("research_agents.agents.github_trending.GITHUB_TRENDING_LANGUAGES", ["python"])
    @patch("research_agents.agents.github_trending.time.sleep")
    def test_dedup_tool_monitor(self, mock_sleep, mock_client, mock_search, mock_exists, store):
        """Should skip repos already seen by tool_monitor."""
        mock_search.return_value = [
            {
                "full_name": "existing/repo",
                "description": "Already tracked by tool_monitor",
                "url": "https://github.com/existing/repo",
                "stars": 500,
                "language": "Python",
                "pushed_at": "2026-03-20",
                "created_at": "2026-03-15",
                "topics": [],
            },
        ]
        # First call (github_trending ID) returns False, second (tool_monitor ID) returns True
        mock_exists.side_effect = [False, True]

        with patch("research_agents.agents.github_trending.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "0 written" in result
        assert "1 skipped (dedup)" in result

    def test_github_token_optional(self):
        """GITHUB_TOKEN should be optional (unauthenticated works at lower rate limit)."""
        from research_agents.agents.github_trending import _get_headers
        import os

        # Without token
        with patch.dict(os.environ, {}, clear=True):
            headers = _get_headers()
            assert "Authorization" not in headers

        # With token
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
            headers = _get_headers()
            assert headers["Authorization"] == "token test-token"
