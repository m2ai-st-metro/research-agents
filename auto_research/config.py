"""AutoResearch experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
AUTO_RESEARCH_ROOT = Path(__file__).resolve().parent
DATA_DIR = AUTO_RESEARCH_ROOT / "data"
EXPERIMENTS_DB = DATA_DIR / "experiments.db"

# --- Ollama ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# --- Experiment Settings ---
# Which agents to experiment on
EXPERIMENT_AGENTS: list[str] = [
    "arxiv",
    "tool_monitor",
    "domain_watch",
    "youtube",
    # "rss" excluded — RSS_FEEDS is a list of {name, url, parser} dicts,
    # not search queries. Mutations change feed names, not search behavior.
    "perplexity",
    # "chatgpt" disabled — ceiling NDR problem (baseline 1.0, can't improve).
    # Replaced by ClaudeClaw scheduled task: Nate Newsletter Digester (task 3cd0ba31).
    "gemini_research",
]

# Agents that cost money to query (for cost tracking, not exclusion)
PAID_QUERY_AGENTS: list[str] = [
    "perplexity",
    "chatgpt",
    "gemini_research",
]

VARIANTS_PER_AGENT = 1  # Number of query variants to test per agent per run
IMPROVEMENT_THRESHOLD = 0.20  # 20% improvement required for auto-commit (raised from 15%)
ROLLBACK_THRESHOLD = 0.10  # 10% weekly drop triggers rollback
AUTO_COMMIT_ENABLED = False  # Set True after validating 5-10 winners manually
MIN_SIGNALS_PER_EXPERIMENT = 2  # Lowered from 5: most agents produce 0.5-4 signals/query/run
MAX_CLAUDE_VALIDATIONS = 3  # Top N winners to validate via Claude API

# Per-agent minimum signal overrides. Data from 1900 signals over 28 days shows
# per-query yields: tool_monitor ~4, arxiv ~1.6, perplexity ~1.8, youtube ~0.5.
# A threshold of 5 would reject every agent except tool_monitor.
MIN_SIGNALS_PER_AGENT: dict[str, int] = {
    "tool_monitor": 3,  # highest yield agent, can afford a higher bar
}

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
