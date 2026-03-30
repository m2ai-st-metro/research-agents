"""Tests for the Reddit scanner agent."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from research_agents.agents.reddit_scanner import (
    _fetch_subreddit_hot,
    _make_signal_id,
    run_agent,
)


class TestMakeSignalId:
    def test_deterministic(self):
        id1 = _make_signal_id("SideProject", "abc123")
        id2 = _make_signal_id("SideProject", "abc123")
        assert id1 == id2

    def test_prefix(self):
        sid = _make_signal_id("SideProject", "abc123")
        assert sid.startswith("reddit-")

    def test_different_inputs(self):
        id1 = _make_signal_id("SideProject", "abc")
        id2 = _make_signal_id("SideProject", "def")
        assert id1 != id2


class TestFetchSubredditHot:
    @patch("research_agents.agents.reddit_scanner.httpx.get")
    def test_parses_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "id": "post1",
                            "title": "Test Post",
                            "selftext": "Body text",
                            "permalink": "/r/SideProject/comments/post1/test/",
                            "score": 42,
                            "num_comments": 5,
                            "stickied": False,
                        }
                    },
                    {
                        "data": {
                            "id": "sticky",
                            "title": "Sticky Post",
                            "selftext": "",
                            "permalink": "/r/SideProject/comments/sticky/",
                            "score": 100,
                            "num_comments": 10,
                            "stickied": True,
                        }
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        posts = _fetch_subreddit_hot("SideProject", limit=10)

        assert len(posts) == 1  # Sticky excluded
        assert posts[0]["post_id"] == "post1"
        assert posts[0]["title"] == "Test Post"
        assert posts[0]["score"] == 42

    @patch("research_agents.agents.reddit_scanner.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.HTTPError("Rate limited")
        posts = _fetch_subreddit_hot("SideProject")
        assert posts == []


class TestRunAgent:
    def test_dry_run(self, store):
        with (
            patch("research_agents.agents.reddit_scanner.get_store", return_value=store),
            patch(
                "research_agents.agents.reddit_scanner._fetch_subreddit_hot",
                return_value=[
                    {
                        "post_id": "p1",
                        "title": "Cool Tool",
                        "selftext": "Check this out",
                        "url": "https://reddit.com/r/SideProject/comments/p1/",
                        "score": 20,
                        "num_comments": 3,
                        "subreddit": "SideProject",
                    }
                ],
            ),
        ):
            result = run_agent(dry_run=True)

        assert "DRY RUN" in result or "new" in result
