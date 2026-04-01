"""Tests for the YouTube trending scanner agent."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Snow-Town contracts are importable
SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource

from research_agents.agents.youtube_scanner import (
    _get_transcript_api,
    _make_signal_id,
    _search_youtube_api,
    _search_youtube_fallback,
    get_transcript,
    run_agent,
    search_youtube,
    summarize_transcript,
)


# --- Sample API responses ---

SAMPLE_YOUTUBE_SEARCH_RESPONSE = {
    "items": [
        {
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "Building Autonomous AI Agents with MCP",
                "description": "Deep dive into Model Context Protocol and autonomous coding agents",
                "channelTitle": "AI Dev Channel",
                "publishedAt": "2026-03-01T10:00:00Z",
                "thumbnails": {
                    "high": {"url": "https://i.ytimg.com/vi/abc123/hq.jpg"}
                },
            },
        },
        {
            "id": {"videoId": "def456"},
            "snippet": {
                "title": "Supply Chain AI: Transforming Logistics",
                "description": "How AI is revolutionizing supply chain management",
                "channelTitle": "Tech Trends",
                "publishedAt": "2026-02-28T15:00:00Z",
                "thumbnails": {
                    "high": {"url": "https://i.ytimg.com/vi/def456/hq.jpg"}
                },
            },
        },
    ]
}

SAMPLE_VIDEO_STATS_RESPONSE = {
    "items": [
        {
            "id": "abc123",
            "statistics": {"viewCount": "150000", "likeCount": "5000"},
            "contentDetails": {"duration": "PT15M30S"},
        },
        {
            "id": "def456",
            "statistics": {"viewCount": "80000", "likeCount": "2000"},
            "contentDetails": {"duration": "PT12M45S"},
        },
    ]
}

SAMPLE_TRANSCRIPT = [
    {"text": "Hello everyone, today we're going to talk about", "start": 0.0},
    {"text": "autonomous AI agents and how they're changing", "start": 3.0},
    {"text": "the way we build software.", "start": 6.0},
    {"text": "The Model Context Protocol, or MCP,", "start": 9.0},
    {"text": "provides a standardized way for LLMs to interact with tools.", "start": 12.0},
]

SAMPLE_SUMMARY_RESPONSE = {
    "summary": "This video explores autonomous AI agents built with MCP. The presenter demonstrates how the Model Context Protocol enables LLMs to interact with external tools in a standardized way, leading to more capable and reliable AI systems.",
    "key_concepts": ["MCP", "autonomous agents", "tool use", "LLM integration"],
    "mermaid_diagram": "graph TD\n    A[LLM] -->|MCP Protocol| B[Tool Server]\n    B --> C[File System]\n    B --> D[Database]\n    B --> E[API]\n    A --> F[Agent Loop]\n    F --> A",
    "tags": ["mcp", "ai-agents", "tool-use", "llm"],
}


# --- Tests ---


class TestMakeSignalId:

    def test_simple_id(self):
        assert _make_signal_id("abc123") == "youtube-abc123"

    def test_id_is_deterministic(self):
        assert _make_signal_id("xyz789") == _make_signal_id("xyz789")


class TestSearchYouTubeApi:

    @patch("research_agents.agents.youtube_scanner._get_video_stats")
    @patch("research_agents.agents.youtube_scanner.httpx.get")
    def test_parse_search_response(self, mock_get, mock_stats):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_YOUTUBE_SEARCH_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mock_stats.return_value = {
            "abc123": {"view_count": 150000, "like_count": 5000, "duration": "PT15M30S"},
            "def456": {"view_count": 80000, "like_count": 2000, "duration": "PT12M45S"},
        }

        videos = _search_youtube_api("AI agents", "fake-key", max_results=5)
        assert len(videos) == 2

        assert videos[0]["video_id"] == "abc123"
        assert videos[0]["title"] == "Building Autonomous AI Agents with MCP"
        assert videos[0]["channel_title"] == "AI Dev Channel"
        assert videos[0]["view_count"] == 150000
        assert videos[0]["url"] == "https://www.youtube.com/watch?v=abc123"

        assert videos[1]["video_id"] == "def456"
        assert videos[1]["title"] == "Supply Chain AI: Transforming Logistics"

    @patch("research_agents.agents.youtube_scanner.httpx.get")
    def test_handles_api_error(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.HTTPError("Connection failed")

        videos = _search_youtube_api("test", "fake-key")
        assert videos == []

    @patch("research_agents.agents.youtube_scanner._get_video_stats")
    @patch("research_agents.agents.youtube_scanner.httpx.get")
    def test_empty_results(self, mock_get, mock_stats):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        mock_stats.return_value = {}

        videos = _search_youtube_api("obscure query", "fake-key")
        assert videos == []


class TestSearchYouTubeFallback:

    @patch("research_agents.agents.youtube_scanner.subprocess.run")
    def test_parse_ytdlp_output(self, mock_run):
        ytdlp_output = json.dumps({
            "id": "vid001",
            "title": "AI Agent Tutorial",
            "description": "Learn about AI agents",
            "channel": "TechChannel",
            "upload_date": "20260301",
            "thumbnail": "https://example.com/thumb.jpg",
            "url": "https://www.youtube.com/watch?v=vid001",
            "view_count": 50000,
            "like_count": 1500,
            "duration_string": "14:30",
        })

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=ytdlp_output,
            stderr="",
        )

        videos = _search_youtube_fallback("AI agents", max_results=1)
        assert len(videos) == 1
        assert videos[0]["video_id"] == "vid001"
        assert videos[0]["title"] == "AI Agent Tutorial"
        assert videos[0]["view_count"] == 50000

    @patch("research_agents.agents.youtube_scanner.subprocess.run")
    def test_handles_ytdlp_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        videos = _search_youtube_fallback("test")
        assert videos == []

    @patch("research_agents.agents.youtube_scanner.subprocess.run")
    def test_handles_ytdlp_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=60)
        videos = _search_youtube_fallback("test")
        assert videos == []

    @patch("research_agents.agents.youtube_scanner.subprocess.run")
    def test_handles_ytdlp_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error occurred",
        )
        videos = _search_youtube_fallback("test")
        assert videos == []


class TestSearchYouTube:

    @patch("research_agents.agents.youtube_scanner._get_youtube_api_key")
    @patch("research_agents.agents.youtube_scanner._search_youtube_api")
    def test_uses_api_when_key_available(self, mock_api, mock_key):
        mock_key.return_value = "test-key"
        mock_api.return_value = [{"video_id": "v1", "title": "Test"}]

        result = search_youtube("test query")
        mock_api.assert_called_once_with("test query", "test-key", 5)
        assert len(result) == 1

    @patch("research_agents.agents.youtube_scanner._get_youtube_api_key")
    @patch("research_agents.agents.youtube_scanner._search_youtube_fallback")
    def test_uses_fallback_when_no_key(self, mock_fallback, mock_key):
        mock_key.return_value = None
        mock_fallback.return_value = [{"video_id": "v2", "title": "Fallback"}]

        result = search_youtube("test query")
        mock_fallback.assert_called_once_with("test query", 5)
        assert len(result) == 1


class TestGetTranscript:

    @patch("research_agents.agents.youtube_scanner._get_transcript_api")
    def test_returns_api_transcript(self, mock_api):
        mock_api.return_value = "This is the transcript text."
        result = get_transcript("abc123")
        assert result == "This is the transcript text."

    @patch("research_agents.agents.youtube_scanner._get_transcript_ytdlp")
    @patch("research_agents.agents.youtube_scanner._get_transcript_api")
    def test_falls_back_to_ytdlp(self, mock_api, mock_ytdlp):
        mock_api.return_value = None
        mock_ytdlp.return_value = "Fallback transcript."
        result = get_transcript("abc123")
        assert result == "Fallback transcript."

    @patch("research_agents.agents.youtube_scanner._get_transcript_ytdlp")
    @patch("research_agents.agents.youtube_scanner._get_transcript_api")
    def test_returns_none_when_both_fail(self, mock_api, mock_ytdlp):
        mock_api.return_value = None
        mock_ytdlp.return_value = None
        result = get_transcript("abc123")
        assert result is None


class TestGetTranscriptApi:

    def test_extracts_transcript(self):
        # v1.x API: YouTubeTranscriptApi().fetch(video_id) returns objects with .text
        entries = [MagicMock(text=e["text"]) for e in SAMPLE_TRANSCRIPT]
        mock_yt_class = MagicMock()
        mock_yt_class.return_value.fetch.return_value = entries
        mock_module = MagicMock()
        mock_module.YouTubeTranscriptApi = mock_yt_class

        with patch.dict("sys.modules", {"youtube_transcript_api": mock_module}):
            result = _get_transcript_api("abc123")

        assert result is not None
        assert "autonomous AI agents" in result
        assert "Model Context Protocol" in result


class TestSummarizeTranscript:
    """Tests for YouTube transcript summarization via Gemini."""

    @patch("research_agents.gemini_client.get_gemini_client")
    def test_summarize_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(SAMPLE_SUMMARY_RESPONSE)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_transcript(
            title="AI Agents with MCP",
            transcript="Sample transcript about AI agents and MCP...",
        )

        assert "summary" in result
        assert "key_concepts" in result
        assert "mermaid_diagram" in result
        assert "MCP" in result["key_concepts"]
        assert "graph TD" in result["mermaid_diagram"]

    @patch("research_agents.gemini_client.get_gemini_client")
    def test_summarize_handles_json_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "not valid json"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_transcript(
            title="Test Video",
            transcript="Test transcript",
        )

        assert "summary" in result
        assert "Test Video" in result["summary"]

    @patch("research_agents.gemini_client.get_gemini_client")
    def test_summarize_handles_markdown_wrapped_json(self, mock_get_client):
        mock_client = MagicMock()
        wrapped = f"```json\n{json.dumps(SAMPLE_SUMMARY_RESPONSE)}\n```"
        mock_response = MagicMock()
        mock_response.text = wrapped
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_transcript(
            title="Test",
            transcript="Test transcript",
        )

        assert "MCP" in result["key_concepts"]


class TestRunAgent:

    @patch("research_agents.agents.youtube_scanner.search_youtube")
    @patch("research_agents.agents.youtube_scanner.YOUTUBE_SEARCH_QUERIES", ["test query"])
    def test_dry_run_no_writes(self, mock_search, store):
        mock_search.return_value = [
            {
                "video_id": "test001",
                "title": "Test Video About AI",
                "description": "A test video",
                "channel_title": "Test Channel",
                "published_at": "2026-03-01T10:00:00Z",
                "url": "https://www.youtube.com/watch?v=test001",
                "view_count": 10000,
                "like_count": 500,
                "duration": "PT10M",
                "thumbnail_url": "",
            }
        ]

        with patch("research_agents.agents.youtube_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "1 new" in result
        assert "0 written" in result
        signals = store.read_signals()
        assert len(signals) == 0

    @patch("research_agents.agents.youtube_scanner.assess_relevance")
    @patch("research_agents.agents.youtube_scanner.summarize_transcript")
    @patch("research_agents.agents.youtube_scanner.get_transcript")
    @patch("research_agents.agents.youtube_scanner.search_youtube")
    @patch("research_agents.agents.youtube_scanner.get_client")
    @patch("research_agents.agents.youtube_scanner.YOUTUBE_SEARCH_QUERIES", ["test query"])
    @patch("research_agents.agents.youtube_scanner.time.sleep")
    def test_full_run_writes_signals(
        self, mock_sleep, mock_client, mock_search, mock_transcript,
        mock_summarize, mock_assess, store
    ):
        mock_search.return_value = [
            {
                "video_id": "vid_high",
                "title": "MCP Agent Architecture Deep Dive",
                "description": "Detailed walkthrough of MCP agent architecture",
                "channel_title": "AI Dev",
                "published_at": "2026-03-01T10:00:00Z",
                "url": "https://www.youtube.com/watch?v=vid_high",
                "view_count": 200000,
                "like_count": 8000,
                "duration": "PT18M",
                "thumbnail_url": "",
            },
            {
                "video_id": "vid_low",
                "title": "Random Cooking Video",
                "description": "How to make pasta",
                "channel_title": "Chef Channel",
                "published_at": "2026-02-28T10:00:00Z",
                "url": "https://www.youtube.com/watch?v=vid_low",
                "view_count": 5000,
                "like_count": 100,
                "duration": "PT8M",
                "thumbnail_url": "",
            },
        ]

        mock_transcript.side_effect = [
            "This is a transcript about MCP and AI agents...",
            "This is about cooking pasta with tomato sauce...",
        ]

        mock_summarize.side_effect = [
            SAMPLE_SUMMARY_RESPONSE,
            {
                "summary": "A cooking tutorial about pasta.",
                "key_concepts": ["pasta", "cooking"],
                "mermaid_diagram": "",
                "tags": ["cooking"],
            },
        ]

        mock_assess.side_effect = [
            {
                "relevance": "high",
                "relevance_rationale": "Directly relevant to MCP development",
                "tags": ["mcp", "agents"],
                "domain": "ai-agents",
            },
            {
                "relevance": "low",
                "relevance_rationale": "Not relevant - cooking content",
                "tags": [],
                "domain": None,
            },
        ]

        with patch("research_agents.agents.youtube_scanner.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        assert "1 skipped" in result

        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].title == "MCP Agent Architecture Deep Dive"
        assert signals[0].source == SignalSource.YOUTUBE_SCANNER
        assert signals[0].relevance == SignalRelevance.HIGH

        # Verify raw_data includes mermaid diagram
        assert signals[0].raw_data is not None
        assert "mermaid_diagram" in signals[0].raw_data
        assert "graph TD" in signals[0].raw_data["mermaid_diagram"]
        assert signals[0].raw_data["video_id"] == "vid_high"
        assert signals[0].raw_data["view_count"] == 200000

    @patch("research_agents.agents.youtube_scanner.assess_relevance")
    @patch("research_agents.agents.youtube_scanner.get_transcript")
    @patch("research_agents.agents.youtube_scanner.search_youtube")
    @patch("research_agents.agents.youtube_scanner.get_client")
    @patch("research_agents.agents.youtube_scanner.YOUTUBE_SEARCH_QUERIES", ["test query"])
    @patch("research_agents.agents.youtube_scanner.time.sleep")
    def test_handles_no_transcript(
        self, mock_sleep, mock_client, mock_search, mock_transcript,
        mock_assess, store
    ):
        """Videos without transcripts should still be assessed based on title/description."""
        mock_search.return_value = [
            {
                "video_id": "no_sub",
                "title": "AI Agent Demo (No Subs)",
                "description": "Cool AI agent demonstration without subtitles",
                "channel_title": "Dev Channel",
                "published_at": "2026-03-01T10:00:00Z",
                "url": "https://www.youtube.com/watch?v=no_sub",
                "view_count": 50000,
                "like_count": 2000,
                "duration": "PT12M",
                "thumbnail_url": "",
            }
        ]

        mock_transcript.return_value = None  # No transcript available

        mock_assess.return_value = {
            "relevance": "medium",
            "relevance_rationale": "Looks relevant based on title",
            "tags": ["ai-agents"],
            "domain": "ai-agents",
        }

        with patch("research_agents.agents.youtube_scanner.get_store", return_value=store):
            result = run_agent(dry_run=False)

        assert "1 written" in result
        assert "1 no transcript" in result

        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].raw_data["has_transcript"] is False

    @patch("research_agents.agents.youtube_scanner.search_youtube")
    @patch("research_agents.agents.youtube_scanner.YOUTUBE_SEARCH_QUERIES", ["test query"])
    def test_dedup_skips_existing(self, mock_search, store):
        """Signals already in the store should be skipped."""
        from research_agents.signal_writer import write_signal
        write_signal(
            signal_id="youtube-existing_vid",
            source=SignalSource.YOUTUBE_SCANNER,
            title="Already exists",
            summary="Already in store",
            relevance=SignalRelevance.HIGH,
            store=store,
        )

        mock_search.return_value = [
            {
                "video_id": "existing_vid",
                "title": "Already exists",
                "description": "Already in store",
                "channel_title": "Test",
                "published_at": "2026-03-01T10:00:00Z",
                "url": "https://www.youtube.com/watch?v=existing_vid",
                "view_count": 10000,
                "like_count": 500,
                "duration": "PT10M",
                "thumbnail_url": "",
            }
        ]

        with patch("research_agents.agents.youtube_scanner.get_store", return_value=store):
            result = run_agent(dry_run=True)

        assert "0 new" in result


class TestMermaidDiagram:
    """Tests to verify Mermaid diagram generation and storage."""

    @patch("research_agents.gemini_client.get_gemini_client")
    def test_mermaid_in_summary_response(self, mock_get_client):
        """Verify that summarize_transcript correctly parses Mermaid diagrams."""
        mock_client = MagicMock()
        response_with_mermaid = {
            "summary": "A video about AI architecture.",
            "key_concepts": ["LLM", "MCP", "agents"],
            "mermaid_diagram": (
                "graph TD\n"
                "    A[User] --> B[Agent]\n"
                "    B --> C[MCP Server]\n"
                "    C --> D[Tools]\n"
                "    B --> E[LLM]\n"
                "    E --> B"
            ),
            "tags": ["architecture", "ai"],
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_with_mermaid)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_transcript(
            title="AI Architecture",
            transcript="Sample transcript",
        )

        assert "graph TD" in result["mermaid_diagram"]
        assert "MCP Server" in result["mermaid_diagram"]
        assert "Agent" in result["mermaid_diagram"]
