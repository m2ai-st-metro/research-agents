"""Research Agents configuration: queries, cadences, model settings."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ST_RECORDS_ROOT = Path(os.environ.get(
    "ST_RECORDS_ROOT",
    str(Path.home() / "projects" / "st-records"),
))

ULTRA_MAGNUS_DB = Path(os.environ.get(
    "IDEA_CATCHER_DB",
    str(Path.home() / "projects" / "ultra-magnus" / "idea-catcher" / "data" / "caught_ideas.db"),
))

IDEAFORGE_DB = Path(os.environ.get(
    "IDEAFORGE_DB",
    str(Path.home() / "projects" / "ideaforge" / "data" / "ideaforge.db"),
))


def get_st_records_db() -> Path:
    """Return path to ST Records persona_metrics.db."""
    return ST_RECORDS_ROOT / "data" / "persona_metrics.db"


# --- Claude API via DeepInfra (used only for idea synthesis) ---
CLAUDE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
CLAUDE_MAX_TOKENS = 4096

# --- Idea Surfacer ---
# Lookback window for signal synthesis. Wider window = more cross-source
# corroboration opportunity. 75-signal cap still in force (see idea_surfacer.py)
# so context never blows; older signals only fill leftover slots after recency
# sort.
IDEA_SURFACER_LOOKBACK_DAYS = int(os.environ.get("IDEA_SURFACER_LOOKBACK_DAYS", "14"))

# --- Ollama (local LLM for relevance assessment + trend reports) ---
# Default: AlienPC GPU (qwen2.5:14b, RTX 5080, ~7s per assessment)
# Fallback: ProBook localhost (qwen2.5:7b-instruct, CPU, ~124s per assessment -- too slow)
# If AlienPC is off, set OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.0.0.35:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# --- Tool/Library Monitor (skill-foundry: MCP ecosystem gaps, agent tooling) ---
TOOL_SEARCH_QUERIES: list[str] = [
    "MCP server reference implementation",
    "MCP bridge service API wrapper",
    "agent skill plugin framework",
    "multi-agent workflow orchestration engine",
    "LLM function calling tool use library",
    "MCP SDK typescript python client binding",
]
TOOL_MAX_RESULTS_PER_QUERY = 10

# --- YouTube Trending Scanner (skill-foundry: agent/MCP tooling content) ---
YOUTUBE_SEARCH_QUERIES: list[str] = [
    "MCP server build tutorial 2026",
    "AI agent framework launch demo 2026",
    "Claude MCP model context application analysis 2026",
    "AI agent workflow architecture pipeline design 2026",
    "AI coding agent review comparison 2026",
]
YOUTUBE_MAX_RESULTS_PER_QUERY = 10
YOUTUBE_MIN_RELEVANCE = "medium"  # Only write signals >= this level
YOUTUBE_TRANSCRIPT_MAX_CHARS = 15000  # Truncate transcripts for summarization
YOUTUBE_API_KEY_ENV = "YOUTUBE_API_KEY"  # env var name for YouTube Data API v3 key

# Channels to always monitor (fetches latest uploads regardless of search queries).
# Supports both channel IDs (UC...) and handles (@name).
YOUTUBE_CHANNELS: list[dict[str, str]] = [
    {"name": "Nate B Jones", "handle": "@natebjones"},
    {"name": "Matt Wolfe", "handle": "@maboroshi"},
    {"name": "AI Jason", "handle": "@AIJasonZ"},
    {"name": "Matthew Berman", "handle": "@matthew_berman"},
    {"name": "David Ondrej", "handle": "@DavidOndrej"},
    {"name": "Fireship", "handle": "@Fireship"},
]
YOUTUBE_CHANNEL_MAX_VIDEOS = 5  # Recent uploads to check per channel per run

# Gemini model for cost-efficient transcript summarization (swapped from Haiku)
YOUTUBE_SUMMARIZER_MODEL = "gemini-3.1-flash-lite-preview"
YOUTUBE_SUMMARIZER_MAX_TOKENS = 4096
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # env var name for Gemini API key

# --- Reddit Scanner (skill-foundry: community pain points, missing tools) ---
REDDIT_SUBREDDITS: list[str] = [
    "devtools",
    "selfhosted",
    "ClaudeAI",
    "LocalLLaMA",
    "ChatGPTPro",
]
REDDIT_POSTS_PER_SUBREDDIT: int = 10
REDDIT_MIN_RELEVANCE: str = "high"  # Raised from medium -- too much noise at medium
REDDIT_MAX_SIGNALS_PER_RUN: int = 15

# --- Cadences (retired agents removed 2026-04-05; doc-drift fixed 2026-04-20) ---
# Values reflect actual user crontab, not aspirational CLAUDE.md claims.
# "planned" = module exists but is not scheduled; must be added to cron
# manually if wanted (keeps paid-API spend off by default).
CADENCE = {
    "tool_monitor": "daily",
    "rss": "daily",
    "gemini_research": "daily",
    "reddit": "daily",
    "youtube": "daily",
    "idea_surfacer": "twice_daily",
    "trend_analyzer": "weekly",
    "perplexity": "planned",
    "chatgpt": "planned",
}

# --- Agent Specializations (anti-monoculture, skill-foundry aligned) ---
# Each LLM agent focuses on a specific signal type to reduce overlap.
AGENT_SPECIALIZATIONS: dict[str, dict[str, str | list[str]]] = {
    "perplexity": {
        "focus": "skill-gaps",
        "description": "MCP ecosystem gaps, missing integrations, new skill/tool releases",
        "do_not_report": [
            "broad market analysis or competitive dynamics",
            "business model speculation",
            "regulatory policy analysis",
            "funding rounds or acquisitions",
        ],
    },
    "chatgpt": {
        "focus": "workflow-patterns",
        "description": (
            "Recurring workflow patterns that lack tooling, "
            "agent skill demand signals, automation gaps"
        ),
        "do_not_report": [
            "specific tool or library releases",
            "individual GitHub repos or open-source projects",
            "technical implementation details",
            "recent news events less than 7 days old",
        ],
    },
    "gemini_research": {
        "focus": "emerging-infra",
        "description": (
            "New MCP servers, agent framework releases (<7 days), "
            "SDK updates, protocol changes"
        ),
        "do_not_report": [
            "established market dynamics or competitive analysis",
            "tools or frameworks that have been available for more than 2 weeks",
            "business model analysis or strategic positioning",
            "broad industry trends without specific recent events",
        ],
    },
}

# --- Persona IDs (for tool_monitor tagging) ---
PERSONA_IDS: list[str] = [
    "carmack",
    "hopper",
    "christensen",
    "porter",
    "knuth",
    "hamilton",
]

# --- RSS/Newsletter Scanner ---
# --- RSS/Newsletter Scanner (skill-foundry: MCP/agent/workflow focused feeds) ---
RSS_FEEDS: list[dict[str, str]] = [
    {
        "name": "Simon Willison",
        "url": "https://simonwillison.net/atom/everything/",
        "parser": "feedparser",
    },
    {"name": "Latent Space", "url": "https://www.latent.space/feed", "parser": "feedparser"},
    {"name": "Dev.to MCP", "url": "https://dev.to/feed/tag/mcp", "parser": "feedparser"},
    {"name": "Dev.to CLI", "url": "https://dev.to/feed/tag/cli", "parser": "feedparser"},
    {"name": "Dev.to Agents", "url": "https://dev.to/feed/tag/agents", "parser": "feedparser"},
    {"name": "Dev.to Automation", "url": "https://dev.to/feed/tag/automation", "parser": "feedparser"},
    {"name": "TLDR AI", "url": "https://tldr.tech/ai/rss", "parser": "feedparser"},
]
RSS_MIN_RELEVANCE = "medium"
RSS_LOOKBACK_DAYS = 3  # Ignore articles older than this

# --- Trend Analyzer ---
TREND_LOOKBACK_DAYS = 14  # Window of signals to analyze
TREND_REPORT_DIR = DATA_DIR / "trend_reports"
TREND_MIN_SIGNALS_FOR_ANALYSIS = 5  # Skip if fewer signals in window
TREND_SUMMARIZER_MODEL = "anthropic/claude-sonnet-4-6"
TREND_SUMMARIZER_MAX_TOKENS = 8192

# --- Firecrawl Enrichment (Test Phase) ---
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"
FIRECRAWL_ENRICHMENT_ENABLED = True  # Toggle to disable all enrichment scrapes
FIRECRAWL_CREDIT_FLOOR = 20  # Stop enriching if credits drop below this
FIRECRAWL_ENRICH_MAX_PER_QUERY = 2  # Max scrapes per search query per run
FIRECRAWL_ENRICH_MAX_CHARS = 3000  # Truncate scraped content for relevance re-assessment

# --- Manual Signal Ingest ---
MANUAL_MIN_RELEVANCE = "low"  # Accept anything -- Matthew curated it

# --- Perplexity Research Agent (routed via OpenRouter) ---
PERPLEXITY_API_KEY_ENV = "OPENROUTER_API_KEY"
PERPLEXITY_MODEL = "perplexity/sonar-pro"
PERPLEXITY_MAX_TOKENS = 4096
PERPLEXITY_RESEARCH_QUERIES: list[str] = [
    "Which MCP servers or Model Context Protocol integrations shipped this week, and what capabilities do they add?",
    "Which agent skills, plugins, or tool-use integrations are developers explicitly requesting that don't yet exist as shipped tools?",
    "Which new workflow or pipeline automation tools for AI agents launched this week?",
    "Which external APIs or SaaS services are developers wrapping as MCP servers or agent tools right now, and what drives the choice?",
    "What specific pain points or missing features are developers actively complaining about in the Claude, ChatGPT, or AI coding agent ecosystems this week?",
]
PERPLEXITY_MIN_RELEVANCE = "medium"

# --- ChatGPT Research Agent ---
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
CHATGPT_MODEL = "gpt-5.4-mini"
CHATGPT_MAX_TOKENS = 4096
CHATGPT_RESEARCH_QUERIES: list[str] = [
    "What recurring workflow patterns do AI agent users need that lack dedicated tooling or MCP servers?",
    "What types of agent skills or automations are developers building repeatedly from scratch instead of using existing tools?",
    "What API integrations are most requested by AI agent framework users but poorly served by existing MCP servers?",
    "Where are the biggest gaps between what AI coding agents can do and what workflow infrastructure supports?",
    "What categories of MCP servers or agent plugins would have the highest utility for solo developers and small teams?",
]
CHATGPT_MIN_RELEVANCE = "medium"

# --- Gemini Research Agent ---
GEMINI_RESEARCH_MODEL = "gemini-3-flash-preview"
GEMINI_RESEARCH_MAX_TOKENS = 4096
GEMINI_RESEARCH_QUERIES: list[str] = [
    "Search for MCP server or Model Context Protocol integration release announcements published in the last 7 days.",
    "Search for AI agent framework, SDK, or orchestration library releases announced in the last 7 days.",
    "Search for GitHub repositories with recent activity related to MCP server implementations or agent skill plugins that have gained popularity in the last week",
    "Search for recent introductions of agent workflow or pipeline automation tools launched within the last week",
    "Search for recent forum debates and developer exchanges within the last week discussing gaps in MCP integrations or unimplemented agent functionalities",
]
GEMINI_RESEARCH_MIN_RELEVANCE = "medium"
