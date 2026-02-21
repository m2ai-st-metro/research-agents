"""Tests for the tool/library monitor agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "snow-town"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.tool_monitor import (
    _make_signal_id,
    _search_github_repos,
    run_agent,
)


SAMPLE_GITHUB_RESPONSE = {
    "items": [
        {
            "full_name": "anthropics/mcp-server-example",
            "description": "Example MCP server for tool augmentation",
            "html_url": "https://github.com/anthropics/mcp-server-example",
            "stargazers_count": 500,
            "language": "Python",
            "pushed_at": "2026-02-20T10:00:00Z",
            "topics": ["mcp", "llm", "tools"],
        },
        {
            "full_name": "some/other-repo",
            "description": "A cooking recipe app",
            "html_url": "https://github.com/some/other-repo",
            "stargazers_count": 10,
            "language": "JavaScript",
            "pushed_at": "2026-02-19T10:00:00Z",
            "topics": [],
        },
    ]
}


class TestMakeSignalId:

    def test_deterministic(self):
        id1 = _make_signal_id("anthropics/mcp-server")
        id2 = _make_signal_id("anthropics/mcp-server")
        assert id1 == id2
        assert id1.startswith("tool-")

    def test_different_inputs(self):
        id1 = _make_signal_id("repo-a")
        id2 = _make_signal_id("repo-b")
        assert id1 != id2


class TestSearchGithub:

    @patch("research_agents.agents.tool_monitor.httpx.get")
    def test_parse_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_GITHUB_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        repos = _search_github_repos("MCP server", max_results=5)
        assert len(repos) == 2
        assert repos[0]["full_name"] == "anthropics/mcp-server-example"
        assert repos[0]["stars"] == 500

    @patch("research_agents.agents.tool_monitor.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.HTTPError("Rate limited")
        repos = _search_github_repos("test")
        assert repos == []


class TestRunAgent:

    @patch("research_agents.agents.tool_monitor._search_github_repos")
    @patch("research_agents.agents.tool_monitor.TOOL_SEARCH_QUERIES", ["test query"])
    def test_dry_run_no_writes(self, mock_search, store):
        mock_search.return_value = [
            {
                "full_name": "test/repo",
                "description": "Test",
                "url": "https://github.com/test/repo",
                "stars": 100,
                "language": "Python",
                "pushed_at": "2026-02-20",
                "topics": [],
            }
        ]

        with patch("research_agents.agents.tool_monitor.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result

    @patch("research_agents.agents.tool_monitor.assess_relevance")
    @patch("research_agents.agents.tool_monitor._search_github_repos")
    @patch("research_agents.agents.tool_monitor.get_client")
    @patch("research_agents.agents.tool_monitor.TOOL_SEARCH_QUERIES", ["test"])
    @patch("research_agents.agents.tool_monitor.time.sleep")
    def test_full_run_writes_signals(self, mock_sleep, mock_client, mock_search, mock_assess, store):
        mock_search.return_value = [
            {
                "full_name": "anthropics/mcp-server",
                "description": "MCP server framework",
                "url": "https://github.com/anthropics/mcp-server",
                "stars": 500,
                "language": "Python",
                "pushed_at": "2026-02-20",
                "topics": ["mcp"],
            },
        ]
        mock_assess.return_value = {
            "relevance": "high",
            "relevance_rationale": "Core MCP tooling",
            "tags": ["mcp"],
            "domain": "developer-tools",
            "persona_tags": ["hopper"],
        }

        with patch("research_agents.agents.tool_monitor.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].source == SignalSource.TOOL_MONITOR
        assert "persona:hopper" in signals[0].tags
