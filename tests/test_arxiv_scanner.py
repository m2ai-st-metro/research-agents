"""Tests for the arXiv scanner agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Snow-Town contracts are importable
ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.arxiv_scanner import (
    _make_signal_id,
    _search_arxiv,
    run_agent,
)


# Sample arXiv API XML response
SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Tool-Augmented LLMs for Code Generation</title>
    <summary>We present a new approach to augmenting large language models with tools for improved code generation.</summary>
    <published>2026-02-20T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345v1" title="pdf" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.67890v1</id>
    <title>MCP Protocol Design Patterns</title>
    <summary>An analysis of design patterns in the Model Context Protocol ecosystem.</summary>
    <published>2026-02-19T00:00:00Z</published>
    <author><name>Carol Lee</name></author>
    <link href="http://arxiv.org/abs/2401.67890v1" rel="alternate" type="text/html"/>
  </entry>
</feed>"""


class TestMakeSignalId:

    def test_simple_id(self):
        assert _make_signal_id("2401.12345v1") == "arxiv-2401.12345v1"

    def test_id_with_slash(self):
        assert _make_signal_id("cs/0401001") == "arxiv-cs-0401001"


class TestSearchArxiv:

    @patch("research_agents.agents.arxiv_scanner.httpx.get")
    def test_parse_arxiv_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ARXIV_XML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = _search_arxiv("test query", max_results=5)
        assert len(papers) == 2

        assert papers[0]["arxiv_id"] == "2401.12345v1"
        assert papers[0]["title"] == "Tool-Augmented LLMs for Code Generation"
        assert "Alice Smith" in papers[0]["authors"]
        assert papers[0]["url"] == "http://arxiv.org/pdf/2401.12345v1"

        assert papers[1]["arxiv_id"] == "2401.67890v1"
        assert papers[1]["title"] == "MCP Protocol Design Patterns"

    @patch("research_agents.agents.arxiv_scanner.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.HTTPError("Connection failed")

        papers = _search_arxiv("test query")
        assert papers == []


class TestRunAgent:

    @patch("research_agents.agents.arxiv_scanner._search_arxiv")
    @patch("research_agents.agents.arxiv_scanner.ARXIV_SEARCH_QUERIES", ["test query"])
    def test_dry_run_no_writes(self, mock_search, store):
        mock_search.return_value = [
            {
                "arxiv_id": "2401.99999v1",
                "title": "Test Paper",
                "summary": "A test paper",
                "url": "http://arxiv.org/abs/2401.99999v1",
                "authors": ["Test Author"],
                "published": "2026-02-20T00:00:00Z",
            }
        ]

        with patch("research_agents.agents.arxiv_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result
        signals = store.read_signals()
        assert len(signals) == 0

    @patch("research_agents.agents.arxiv_scanner.assess_relevance")
    @patch("research_agents.agents.arxiv_scanner._search_arxiv")
    @patch("research_agents.agents.arxiv_scanner.get_client")
    @patch("research_agents.agents.arxiv_scanner.ARXIV_SEARCH_QUERIES", ["test query"])
    @patch("research_agents.agents.arxiv_scanner.time.sleep")
    def test_full_run_writes_signals(self, mock_sleep, mock_client, mock_search, mock_assess, store):
        mock_search.return_value = [
            {
                "arxiv_id": "2401.11111v1",
                "title": "Relevant Paper",
                "summary": "Very relevant to MCP development",
                "url": "http://arxiv.org/abs/2401.11111v1",
                "authors": ["Author A"],
                "published": "2026-02-20T00:00:00Z",
            },
            {
                "arxiv_id": "2401.22222v1",
                "title": "Irrelevant Paper",
                "summary": "About cooking recipes",
                "url": "http://arxiv.org/abs/2401.22222v1",
                "authors": ["Author B"],
                "published": "2026-02-19T00:00:00Z",
            },
        ]

        mock_assess.side_effect = [
            {
                "relevance": "high",
                "relevance_rationale": "Directly relevant to MCP work",
                "tags": ["mcp", "agents"],
                "domain": "ai-agents",
                "persona_tags": ["carmack"],
            },
            {
                "relevance": "low",
                "relevance_rationale": "Not relevant",
                "tags": [],
                "domain": None,
                "persona_tags": [],
            },
        ]

        with patch("research_agents.agents.arxiv_scanner.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        assert "1 skipped" in result

        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].title == "Relevant Paper"
        assert signals[0].relevance == SignalRelevance.HIGH
        assert "persona:carmack" in signals[0].tags

    @patch("research_agents.agents.arxiv_scanner._search_arxiv")
    @patch("research_agents.agents.arxiv_scanner.ARXIV_SEARCH_QUERIES", ["test query"])
    def test_dedup_skips_existing(self, mock_search, store):
        """Signals already in the store should be skipped."""
        # Pre-populate store with an existing signal
        from research_agents.signal_writer import write_signal
        write_signal(
            signal_id="arxiv-2401.11111v1",
            source=SignalSource.ARXIV_HF,
            title="Already exists",
            summary="Already in store",
            relevance=SignalRelevance.HIGH,
            store=store,
        )

        mock_search.return_value = [
            {
                "arxiv_id": "2401.11111v1",
                "title": "Already exists",
                "summary": "Already in store",
                "url": "http://arxiv.org/abs/2401.11111v1",
                "authors": ["Author"],
                "published": "2026-02-20T00:00:00Z",
            }
        ]

        with patch("research_agents.agents.arxiv_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "0 new" in result
