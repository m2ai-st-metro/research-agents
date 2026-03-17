"""AutoResearch experiment configuration."""

from __future__ import annotations

from pathlib import Path

# --- Paths ---
AUTO_RESEARCH_ROOT = Path(__file__).resolve().parent
DATA_DIR = AUTO_RESEARCH_ROOT / "data"
EXPERIMENTS_DB = DATA_DIR / "experiments.db"

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT = 120  # seconds per request

# --- Experiment Settings ---
# Which agents to experiment on (must have free API sources)
EXPERIMENT_AGENTS: list[str] = [
    "arxiv",
    "tool_monitor",
    "domain_watch",
    "youtube",
    # "rss" excluded — RSS_FEEDS is a list of {name, url, parser} dicts,
    # not search queries. Mutations change feed names, not search behavior.
]

# Agents that cost money to query (excluded from default experiments)
PAID_QUERY_AGENTS: list[str] = [
    "perplexity",
    "chatgpt",
    "gemini_research",
]

VARIANTS_PER_AGENT = 1  # Number of query variants to test per agent per run
IMPROVEMENT_THRESHOLD = 0.15  # 15% improvement required for auto-commit
ROLLBACK_THRESHOLD = 0.10  # 10% weekly drop triggers rollback
MIN_SIGNALS_PER_EXPERIMENT = 5  # Minimum signals for valid comparison
MAX_CLAUDE_VALIDATIONS = 3  # Top N winners to validate via Claude API

# --- Scoring (mirrors IdeaForge config) ---
SCORE_WEIGHTS: dict[str, float] = {
    "opportunity": 0.25,
    "problem": 0.25,
    "feasibility": 0.20,
    "why_now": 0.15,
    "competition": 0.15,
}

DISMISS_THRESHOLD = 4.5
ROUTE_THRESHOLDS: dict[str, float] = {
    "product": 6.0,
    "agent": 5.0,
    "tool": 4.5,
}

# --- Query Config Mapping ---
# Maps agent name -> config attribute holding its queries
AGENT_QUERY_KEYS: dict[str, str] = {
    "arxiv": "ARXIV_SEARCH_QUERIES",
    "tool_monitor": "TOOL_SEARCH_QUERIES",
    "domain_watch": "DOMAIN_WATCH_QUERIES",
    "youtube": "YOUTUBE_SEARCH_QUERIES",
    "rss": "RSS_FEEDS",
    "perplexity": "PERPLEXITY_RESEARCH_QUERIES",
    "chatgpt": "CHATGPT_RESEARCH_QUERIES",
    "gemini_research": "GEMINI_RESEARCH_QUERIES",
}
