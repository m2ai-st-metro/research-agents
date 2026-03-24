"""Research Agents configuration: queries, cadences, model settings."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SNOW_TOWN_ROOT = Path(os.environ.get(
    "SNOW_TOWN_ROOT",
    str(Path.home() / "projects" / "st-factory"),
))

ULTRA_MAGNUS_DB = Path(os.environ.get(
    "IDEA_CATCHER_DB",
    str(Path.home() / "projects" / "ultra-magnus" / "idea-catcher" / "data" / "caught_ideas.db"),
))

IDEAFORGE_DB = Path(os.environ.get(
    "IDEAFORGE_DB",
    str(Path.home() / "projects" / "ideaforge" / "data" / "ideaforge.db"),
))


def get_snow_town_db() -> Path:
    """Return path to Snow-Town persona_metrics.db."""
    return SNOW_TOWN_ROOT / "data" / "persona_metrics.db"


# --- Claude API via DeepInfra (used only for idea synthesis) ---
CLAUDE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
CLAUDE_MAX_TOKENS = 4096

# --- Ollama (local LLM for relevance assessment + trend reports) ---
# Default: ProBook localhost (qwen2.5:7b-instruct, CPU, always available)
# Override via env for AlienPC: OLLAMA_BASE_URL=http://10.0.0.35:11434 OLLAMA_MODEL=qwen2.5:14b
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))  # Higher for CPU inference

# --- ArXiv / Paper Scanner ---
ARXIV_SEARCH_QUERIES: list[str] = [
    "interactive neural design",
    "code generation autonomous systems",
    "prompt tuning data analysis",
    "LLM integration challenges",
    "code generation efficiency optimization",
    "LLM performance scaling",
    "automated workflow systems analysis",
]
ARXIV_MAX_RESULTS_PER_QUERY = 10
ARXIV_MIN_RELEVANCE = "medium"  # Only write signals >= this level

# --- Tool/Library Monitor ---
TOOL_SEARCH_QUERIES: list[str] = [
    "MCP architecture repository",
    "Claude API",
    "AI coding assistant",
    "LLM deployment infrastructure",
    "prompt engineering framework",
    "AI agent development strategies",
]
TOOL_MAX_RESULTS_PER_QUERY = 10

# --- Domain Watcher ---
DOMAIN_WATCH_QUERIES: list[str] = [
    "healthcare AI home health",
    "solo developer AI tools",
    "automated workflow AI solutions",
    "HIPAA compliant AI",
    "clinical AI diagnostic tools",
]
DOMAIN_MIN_RELEVANCE = "high"  # Higher bar for adjacent domains

# --- YouTube Trending Scanner ---
YOUTUBE_SEARCH_QUERIES: list[str] = [
    "autonomous coding assistants in AI development 2023",
    "supply chain AI automation",
    "MCP model applications in AI-driven workflow automation 2023",
    "LLM integration for enhancing AI development environments 2023",
    "AI developer tools 2026",
    "healthcare AI technology",
    "agentic systems integration in ai workflows 2023",
]
YOUTUBE_MAX_RESULTS_PER_QUERY = 5
YOUTUBE_MIN_RELEVANCE = "medium"  # Only write signals >= this level
YOUTUBE_TRANSCRIPT_MAX_CHARS = 15000  # Truncate transcripts for summarization
YOUTUBE_API_KEY_ENV = "YOUTUBE_API_KEY"  # env var name for YouTube Data API v3 key

# Channels to always monitor (fetches latest uploads regardless of search queries).
# Supports both channel IDs (UC...) and handles (@name).
YOUTUBE_CHANNELS: list[dict[str, str]] = [
    {"name": "Nate B Jones", "handle": "@natebjones"},
]
YOUTUBE_CHANNEL_MAX_VIDEOS = 5  # Recent uploads to check per channel per run

# Gemini model for cost-efficient transcript summarization (swapped from Haiku)
YOUTUBE_SUMMARIZER_MODEL = "gemini-3.1-flash-lite-preview"
YOUTUBE_SUMMARIZER_MAX_TOKENS = 4096
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # env var name for Gemini API key

# --- Reddit Scanner ---
REDDIT_SUBREDDITS: list[str] = [
    "SideProject",
    "indiehackers",
    "selfhosted",
    "devtools",
    "MachineLearning",
]
REDDIT_POSTS_PER_SUBREDDIT: int = 10
REDDIT_MIN_RELEVANCE: str = "medium"
REDDIT_MAX_SIGNALS_PER_RUN: int = 15

# --- Product Hunt Scanner ---
PRODUCTHUNT_RSS_URL: str = "https://www.producthunt.com/feed"
PRODUCTHUNT_MAX_ITEMS: int = 20
PRODUCTHUNT_MIN_RELEVANCE: str = "medium"
PRODUCTHUNT_MAX_SIGNALS_PER_RUN: int = 10

# --- Cadences ---
CADENCE = {
    "arxiv": "daily",
    "tool_monitor": "daily",
    "domain_watch": "every_3_days",
    "idea_surfacer": "daily",
    "youtube": "daily",
    "rss": "daily",
    "trend_analyzer": "weekly",
    "perplexity": "daily",
    "chatgpt": "every_3_days",
    "gemini_research": "daily",
    "github_trending": "daily",
    "reddit": "daily",
    "product_hunt": "daily",
}

# --- Agent Specializations (anti-monoculture) ---
# Each LLM agent focuses on a specific signal type to reduce overlap.
AGENT_SPECIALIZATIONS: dict[str, dict[str, str | list[str]]] = {
    "perplexity": {
        "focus": "tactical",
        "description": "Tool launches, releases, MCP ecosystem, specific product updates",
        "do_not_report": [
            "broad market analysis or competitive dynamics",
            "business model speculation",
            "regulatory policy analysis",
            "funding rounds or acquisitions",
        ],
    },
    "chatgpt": {
        "focus": "strategic",
        "description": (
            "Market dynamics, competitive positioning, "
            "business models, underserved niches"
        ),
        "do_not_report": [
            "specific tool or library releases",
            "individual GitHub repos or open-source projects",
            "technical implementation details",
            "recent news events less than 7 days old",
        ],
    },
    "gemini_research": {
        "focus": "emerging",
        "description": (
            "Recent developments (<7 days), regulatory changes, "
            "funding rounds, acquisitions"
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
RSS_FEEDS: list[dict[str, str]] = [
    {"name": "Import AI", "url": "https://jack-clark.net/feed/", "parser": "feedparser"},
    {"name": "TLDR AI", "url": "https://tldr.tech/ai/rss", "parser": "feedparser"},
    {"name": "Latent Space", "url": "https://www.latent.space/feed", "parser": "feedparser"},
    {
        "name": "Simon Willison",
        "url": "https://simonwillison.net/atom/everything/",
        "parser": "feedparser",
    },
    {"name": "The Batch", "url": "https://www.deeplearning.ai/the-batch/", "parser": "firecrawl"},
]
RSS_MIN_RELEVANCE = "medium"
RSS_LOOKBACK_DAYS = 3  # Ignore articles older than this

# --- Trend Analyzer ---
TREND_LOOKBACK_DAYS = 14  # Window of signals to analyze
TREND_REPORT_DIR = DATA_DIR / "trend_reports"
TREND_MIN_SIGNALS_FOR_ANALYSIS = 5  # Skip if fewer signals in window
TREND_SUMMARIZER_MODEL = "anthropic/claude-4-sonnet"
TREND_SUMMARIZER_MAX_TOKENS = 8192

# --- Firecrawl Enrichment (Test Phase) ---
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"
FIRECRAWL_ENRICHMENT_ENABLED = True  # Toggle to disable all enrichment scrapes
FIRECRAWL_CREDIT_FLOOR = 20  # Stop enriching if credits drop below this
FIRECRAWL_ENRICH_MAX_PER_QUERY = 2  # Max scrapes per search query per run
FIRECRAWL_ENRICH_MAX_CHARS = 3000  # Truncate scraped content for relevance re-assessment

# --- Manual Signal Ingest ---
MANUAL_MIN_RELEVANCE = "low"  # Accept anything -- Matthew curated it

# --- Perplexity Research Agent ---
PERPLEXITY_API_KEY_ENV = "PERPLEXITY_API_KEY"
PERPLEXITY_MODEL = "sonar-pro"
PERPLEXITY_MAX_TOKENS = 4096
PERPLEXITY_RESEARCH_QUERIES: list[str] = [
    "What are the biggest emerging pain points for solo AI developers this week?",
    "What recent developments in AI agent technologies and server infrastructure are impacting solo developers?",
    "What healthcare AI regulatory changes or HIPAA-compliant tools appeared this week?",
    "What are the latest trends in autonomous coding agents and AI-assisted software engineering?",
    "What workflow automation tools are gaining traction among small teams?",
]
PERPLEXITY_MIN_RELEVANCE = "medium"

# --- ChatGPT Research Agent ---
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
CHATGPT_MODEL = "gpt-5.4"
CHATGPT_MAX_TOKENS = 4096
CHATGPT_RESEARCH_QUERIES: list[str] = [
    "Analyze current market gaps in AI developer tooling for solo practitioners and small consultancies.",
    "What underserved niches exist at the intersection of healthcare AI and home health services?",
    "What are the most promising business models for AI-powered SaaS products targeting developers?",
    "Identify emerging competitive dynamics in the autonomous coding agent space.",
    "What infrastructure or platform plays are emerging around MCP and tool-augmented LLMs?",
]
CHATGPT_MIN_RELEVANCE = "medium"

# --- Gemini Research Agent ---
GEMINI_RESEARCH_MODEL = "gemini-3.1-pro-preview"
GEMINI_RESEARCH_MAX_TOKENS = 4096
GEMINI_RESEARCH_QUERIES: list[str] = [
    "Search for recent announcements about AI agent frameworks, MCP servers, or tool-use APIs.",
    "Search for emerging healthcare AI startups or products focused on home health and HIPAA compliance.",
    "Search for recent AI-driven enhancements in developer tools and platforms launched within the last week",
    "Search for recent funding rounds or acquisitions in the AI developer tools space.",
    "Search for open-source AI projects gaining traction on GitHub related to autonomous coding.",
]
GEMINI_RESEARCH_MIN_RELEVANCE = "medium"
