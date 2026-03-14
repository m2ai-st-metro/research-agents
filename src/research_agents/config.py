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


# --- Claude API ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_MAX_TOKENS = 4096

# --- ArXiv / Paper Scanner ---
ARXIV_SEARCH_QUERIES: list[str] = [
    "MCP model context protocol",
    "autonomous coding agents",
    "prompt engineering techniques",
    "tool-augmented LLM",
    "AI code generation evaluation",
    "LLM self-improvement",
    "agentic workflow orchestration",
]
ARXIV_MAX_RESULTS_PER_QUERY = 10
ARXIV_MIN_RELEVANCE = "medium"  # Only write signals >= this level

# --- Tool/Library Monitor ---
TOOL_SEARCH_QUERIES: list[str] = [
    "MCP server",
    "Claude API",
    "AI coding assistant",
    "LLM framework",
    "prompt management tool",
    "AI agent framework",
]
TOOL_MAX_RESULTS_PER_QUERY = 10

# --- Domain Watcher ---
DOMAIN_WATCH_QUERIES: list[str] = [
    "healthcare AI home health",
    "solo developer AI tools",
    "workflow automation AI",
    "HIPAA compliant AI",
    "clinical decision support",
]
DOMAIN_MIN_RELEVANCE = "high"  # Higher bar for adjacent domains

# --- Perplexity Web Research ---
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"
PERPLEXITY_SEARCH_QUERIES: list[str] = [
    "latest MCP model context protocol servers and tooling 2026",
    "Claude API new features and capabilities 2026",
    "AI coding agent benchmarks and evaluations",
    "autonomous software development pipelines",
    "LLM self-improvement and meta-learning research",
    "healthcare AI home health regulatory updates",
    "solo developer AI-augmented workflows",
]
PERPLEXITY_MAX_RESULTS_PER_QUERY = 5
PERPLEXITY_MIN_RELEVANCE = "medium"

# --- Cadences ---
CADENCE = {
    "arxiv": "daily",
    "tool_monitor": "daily",
    "domain_watch": "every_3_days",
    "perplexity": "daily",
    "idea_surfacer": "weekly",
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
