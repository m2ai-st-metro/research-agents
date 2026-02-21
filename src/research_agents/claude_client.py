"""Shared Claude API wrapper for relevance assessment."""

from __future__ import annotations

import json
import logging
import os

from anthropic import Anthropic

from .config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger(__name__)


def get_client() -> Anthropic:
    """Create an Anthropic client from environment."""
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def assess_relevance(
    title: str,
    summary: str,
    source_context: str,
    client: Anthropic | None = None,
    model: str = CLAUDE_MODEL,
) -> dict:
    """Assess the relevance of a research signal to the Snow-Town ecosystem.

    Returns dict with:
        relevance: "high" | "medium" | "low"
        relevance_rationale: str
        tags: list[str]
        domain: str | None
        persona_tags: list[str]  (persona IDs this is relevant to)
    """
    if client is None:
        client = get_client()

    prompt = f"""Assess the relevance of this research signal to a solo AI developer's ecosystem.

The developer builds:
- Claude-powered MCP servers and tool-augmented agents
- An autonomous idea-to-product pipeline (Ultra Magnus)
- A self-improving feedback loop (Snow-Town: UM -> Sky-Lynx -> Academy)
- Healthcare AI projects (HIPAA-compliant, home health focus)
- Developer productivity tools

Signal:
- Title: {title}
- Summary: {summary}
- Source context: {source_context}

Respond with JSON only:
{{
    "relevance": "high" | "medium" | "low",
    "relevance_rationale": "Why this is/isn't relevant (1-2 sentences)",
    "tags": ["tag1", "tag2"],
    "domain": "primary domain (e.g. ai-agents, healthcare-ai, developer-tools, etc.) or null",
    "persona_tags": ["persona_id1"] // from: carmack, hopper, christensen, porter, knuth, hamilton — only if directly relevant
}}"""

    response = client.messages.create(
        model=model,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse relevance response: {text[:200]}")
        result = {
            "relevance": "low",
            "relevance_rationale": "Failed to parse assessment",
            "tags": [],
            "domain": None,
            "persona_tags": [],
        }

    return result
