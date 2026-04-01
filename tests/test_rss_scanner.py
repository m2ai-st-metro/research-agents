"""Tests for the RSS scanner agent."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Snow-Town contracts are importable
SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.rss_scanner import (
    _fetch_via_feedparser,
    _is_within_lookback,
    _make_signal_id,
    run_agent,
)


class TestMakeSignalId:

    def test_consistent_id(self):
        id1 = _make_signal_id("https://example.com/article/1")
        id2 = _make_signal_id("https://example.com/article/1")
        assert id1 == id2
        assert id1.startswith("rss-")
        assert len(id1) == 4 + 12  # "rss-" + 12 hex chars

    def test_different_urls_different_ids(self):
        id1 = _make_signal_id("https://example.com/a")
        id2 = _make_signal_id("https://example.com/b")
        assert id1 != id2


class TestFetchViaFeedparser:

    @patch("research_agents.agents.rss_scanner.feedparser.parse")
    def test_parse_entries(self, mock_parse):
        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda k, d="": {
            "title": "AI Newsletter Article",
            "link": "https://example.com/1",
            "summary": "Summary of the article",
            "description": "",
            "published_parsed": time.struct_time((2026, 3, 4, 0, 0, 0, 0, 63, 0)),
        }.get(k, d)

        mock_result = MagicMock()
        mock_result.entries = [mock_entry]
        mock_parse.return_value = mock_result

        articles = _fetch_via_feedparser({"name": "Test Feed", "url": "https://example.com/feed"})
        assert len(articles) == 1
        assert articles[0]["feed_name"] == "Test Feed"

    @patch("research_agents.agents.rss_scanner.feedparser.parse")
    def test_handles_empty_feed(self, mock_parse):
        mock_result = MagicMock()
        mock_result.entries = []
        mock_parse.return_value = mock_result

        articles = _fetch_via_feedparser({"name": "Empty", "url": "https://x.com/feed"})
        assert articles == []


class TestIsWithinLookback:

    def test_no_date_assumed_recent(self):
        """Articles with no published date should be treated as recent."""
        article: dict[str, object] = {"published": None}
        assert _is_within_lookback(article, days=3) is True

    def test_old_article_fails(self):
        article: dict[str, object] = {
            "published": time.struct_time((2020, 1, 1, 0, 0, 0, 0, 1, 0))
        }
        assert _is_within_lookback(article, days=3) is False

    def test_recent_article_passes(self):
        now = time.gmtime()
        article: dict[str, object] = {"published": now}
        assert _is_within_lookback(article, days=3) is True


class TestRunAgent:

    @patch("research_agents.agents.rss_scanner._fetch_via_feedparser")
    @patch("research_agents.agents.rss_scanner.RSS_FEEDS", [{"name": "Test", "url": "x", "parser": "feedparser"}])
    @patch("research_agents.agents.rss_scanner.time.sleep")
    def test_dry_run_no_writes(self, mock_sleep, mock_fetch, store):
        mock_fetch.return_value = [
            {
                "title": "Test Article",
                "url": "https://example.com/1",
                "summary": "Summary",
                "published": None,
                "feed_name": "Test",
            }
        ]

        with patch("research_agents.agents.rss_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result

    @patch("research_agents.agents.rss_scanner.assess_relevance")
    @patch("research_agents.agents.rss_scanner._fetch_via_feedparser")
    @patch("research_agents.agents.rss_scanner.get_client")
    @patch("research_agents.agents.rss_scanner.RSS_FEEDS", [{"name": "Test", "url": "x", "parser": "feedparser"}])
    @patch("research_agents.agents.rss_scanner.time.sleep")
    def test_full_run_writes_signals(self, mock_sleep, mock_client, mock_fetch, mock_assess, store):
        mock_fetch.return_value = [
            {
                "title": "Relevant Article",
                "url": "https://example.com/relevant",
                "summary": "AI agents article",
                "published": None,
                "feed_name": "Test",
            }
        ]
        mock_assess.return_value = {
            "relevance": "high",
            "relevance_rationale": "Directly relevant",
            "tags": ["mcp", "agents"],
            "domain": "ai-agents",
        }

        with patch("research_agents.agents.rss_scanner.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].source == SignalSource.RSS_SCANNER

    @patch("research_agents.agents.rss_scanner._fetch_via_feedparser")
    @patch("research_agents.agents.rss_scanner.RSS_FEEDS", [{"name": "Test", "url": "x", "parser": "feedparser"}])
    @patch("research_agents.agents.rss_scanner.time.sleep")
    def test_dedup_skips_existing(self, mock_sleep, mock_fetch, store):
        from research_agents.signal_writer import write_signal

        # Pre-populate with existing signal
        write_signal(
            signal_id=_make_signal_id("https://example.com/existing"),
            source=SignalSource.RSS_SCANNER,
            title="Already exists",
            summary="Already in store",
            relevance=SignalRelevance.MEDIUM,
            store=store,
        )

        mock_fetch.return_value = [
            {
                "title": "Already exists",
                "url": "https://example.com/existing",
                "summary": "Already in store",
                "published": None,
                "feed_name": "Test",
            }
        ]

        with patch("research_agents.agents.rss_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "0 new" in result
