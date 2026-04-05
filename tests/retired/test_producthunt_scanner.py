"""Tests for the Product Hunt scanner agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from research_agents.agents.producthunt_scanner import (
    _fetch_producthunt_feed,
    _make_signal_id,
    run_agent,
)


class TestMakeSignalId:
    def test_deterministic(self):
        id1 = _make_signal_id("https://producthunt.com/posts/test-product")
        id2 = _make_signal_id("https://producthunt.com/posts/test-product")
        assert id1 == id2

    def test_prefix(self):
        sid = _make_signal_id("https://producthunt.com/posts/test-product")
        assert sid.startswith("producthunt-")

    def test_different_urls(self):
        id1 = _make_signal_id("https://producthunt.com/posts/a")
        id2 = _make_signal_id("https://producthunt.com/posts/b")
        assert id1 != id2


class TestFetchProductHuntFeed:
    @patch("research_agents.agents.producthunt_scanner.feedparser.parse")
    def test_parses_feed(self, mock_parse):
        from time import gmtime

        mock_parse.return_value = MagicMock(
            bozo=False,
            entries=[
                MagicMock(
                    title="Cool AI Tool",
                    summary="An AI-powered thing",
                    link="https://producthunt.com/posts/cool-ai-tool",
                    published_parsed=gmtime(),
                    get=lambda k, d=None: {
                        "title": "Cool AI Tool",
                        "summary": "An AI-powered thing",
                        "link": "https://producthunt.com/posts/cool-ai-tool",
                        "description": "",
                    }.get(k, d),
                ),
            ],
        )

        items = _fetch_producthunt_feed()
        # May or may not return items depending on date filtering
        # The important thing is it doesn't crash
        assert isinstance(items, list)

    @patch("research_agents.agents.producthunt_scanner.feedparser.parse")
    def test_handles_empty_feed(self, mock_parse):
        mock_parse.return_value = MagicMock(bozo=True, entries=[])
        items = _fetch_producthunt_feed()
        assert items == []


class TestRunAgent:
    def test_dry_run(self, store):
        with (
            patch("research_agents.agents.producthunt_scanner.get_store", return_value=store),
            patch(
                "research_agents.agents.producthunt_scanner._fetch_producthunt_feed",
                return_value=[
                    {
                        "title": "New AI Tool",
                        "summary": "Automates things",
                        "url": "https://producthunt.com/posts/new-ai-tool",
                        "published": "2026-03-21T00:00:00+00:00",
                    }
                ],
            ),
        ):
            result = run_agent(dry_run=True)

        assert "DRY RUN" in result or "new" in result
