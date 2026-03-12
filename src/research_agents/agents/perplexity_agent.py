"""Perplexity Research Agent.

Uses Perplexity Sonar API with web search grounding to discover signals
with citations. Anti-monoculture: Perplexity has its own search index
and ranking, producing different signals than Claude or Gemini.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

from ..config import (
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_MAX_TOKENS,
    PERPLEXITY_MIN_RELEVANCE,
    PERPLEXITY_MODEL,
    PERPLEXITY_RESEARCH_QUERIES,
)
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}

SYSTEM_PROMPT = """You are a research analyst for a solo AI developer/consultant.
Your job: identify actionable signals -- emerging tools, market shifts, regulatory
changes, pain points, or opportunities.

The developer's focus areas:
- Claude-powered MCP servers and tool-augmented agents
- Autonomous idea-to-product pipelines
- Healthcare AI (HIPAA-compliant, home health)
- Developer productivity tools for solo practitioners
- Workflow automation for small teams

For each query, return 2-5 discrete signals as JSON:
{
    "signals": [
        {
            "title": "Short descriptive title",
            "summary": "2-3 sentence summary of the signal and why it matters",
            "url": "Source URL (from your citations if available, or null)",
            "relevance": "high" | "medium" | "low",
            "relevance_rationale": "Why this matters to the developer (1 sentence)",
            "tags": ["tag1", "tag2"],
            "domain": "primary domain (ai-agents, healthcare-ai, developer-tools, etc.)"
        }
    ]
}

Only return signals that are genuinely new or noteworthy. Skip obvious/stale info."""


def _make_signal_id(query: str, title: str) -> str:
    """Generate deterministic signal ID from query + title."""
    key = f"pplx-{query[:50]}-{title[:80]}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"perplexity-{digest}"


def _call_perplexity(query: str, api_key: str) -> dict[str, Any]:
    """Call Perplexity Sonar API and return parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": PERPLEXITY_MAX_TOKENS,
    }

    resp = httpx.post(
        PERPLEXITY_API_URL,
        headers=headers,
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()

    # Extract citations if available
    citations = data.get("citations", [])

    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Perplexity response: %s", text[:200])
        result = {"signals": []}

    # Attach citations metadata
    result["_citations"] = citations
    return result


def run_agent(dry_run: bool = False) -> str:
    """Run the Perplexity research agent.

    1. Send each research query to Perplexity Sonar API
    2. Parse structured signal responses
    3. Deduplicate against existing signals
    4. Write signals that meet the relevance bar

    Returns summary string.
    """
    api_key = os.environ.get(PERPLEXITY_API_KEY_ENV)
    if not api_key:
        return f"Skipped: missing env var {PERPLEXITY_API_KEY_ENV}"

    store = get_store()

    total_signals = 0
    total_new = 0
    total_written = 0
    skipped_below_bar = 0

    try:
        for query in PERPLEXITY_RESEARCH_QUERIES:
            logger.info(f"Perplexity query: '{query[:60]}...'")

            if dry_run:
                logger.info(f"  [DRY RUN] Would query Perplexity: {query[:60]}")
                continue

            try:
                result = _call_perplexity(query, api_key)
            except httpx.HTTPError as e:
                logger.warning(f"Perplexity API error for query: {e}")
                continue

            signals = result.get("signals", [])
            citations = result.get("_citations", [])
            total_signals += len(signals)

            for sig in signals:
                title = sig.get("title", "Untitled")
                signal_id = _make_signal_id(query, title)

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1
                relevance = sig.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(PERPLEXITY_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_below_bar += 1
                    logger.debug(f"  Skipped (below bar): {title[:60]}")
                    continue

                # Use signal's own URL, or first citation, or None
                url = sig.get("url")
                if not url and citations:
                    url = citations[0] if isinstance(citations[0], str) else None

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.PERPLEXITY,
                    title=title,
                    summary=sig.get("summary", ""),
                    url=url,
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=sig.get("relevance_rationale", ""),
                    tags=sig.get("tags", []),
                    domain=sig.get("domain"),
                    raw_data={
                        "query": query,
                        "model": PERPLEXITY_MODEL,
                        "citations": citations,
                    },
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {title[:60]}")

            # Rate limit between queries
            time.sleep(2.0)

    finally:
        store.close()

    return (
        f"Perplexity: {total_signals} signals from {len(PERPLEXITY_RESEARCH_QUERIES)} queries, "
        f"{total_new} new, {total_written} written, {skipped_below_bar} skipped (below bar)"
    )
