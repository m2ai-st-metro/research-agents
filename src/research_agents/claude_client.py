"""Shared API wrappers for research-agents pipeline.

- Relevance assessment: Ollama (local, no API cost)
- Idea synthesis: Anthropic Claude (kept on Sonnet for quality)
"""

from __future__ import annotations

import logging
import os

from anthropic import Anthropic

from .config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from .ollama_client import assess_relevance_ollama

logger = logging.getLogger(__name__)


def get_client() -> Anthropic:
    """Create an Anthropic client from environment.

    Used only by idea_surfacer for synthesis (the one task
    where frontier model quality matters).
    """
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def assess_relevance(
    title: str,
    summary: str,
    source_context: str,
    client=None,
    model: str = CLAUDE_MODEL,
) -> dict:
    """Assess the relevance of a research signal.

    Routes to local Ollama instead of Anthropic API.
    The client and model params are kept for interface compatibility
    but are ignored — Ollama handles everything.

    Returns dict with:
        relevance: "high" | "medium" | "low"
        relevance_rationale: str
        tags: list[str]
        domain: str | None
        persona_tags: list[str]
    """
    return assess_relevance_ollama(
        title=title,
        summary=summary,
        source_context=source_context,
    )
