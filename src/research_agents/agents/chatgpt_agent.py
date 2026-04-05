"""ChatGPT Research Agent.

Uses OpenAI GPT-4o for market analysis and strategic signal generation.
Anti-monoculture: GPT-4o has different training data, knowledge cutoff,
and reasoning patterns from Claude, producing complementary signals.
Runs every 3 days (strategic analysis doesn't need daily cadence).
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
    CHATGPT_MAX_TOKENS,
    CHATGPT_MIN_RELEVANCE,
    CHATGPT_MODEL,
    CHATGPT_RESEARCH_QUERIES,
    OPENAI_API_KEY_ENV,
)
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}

SYSTEM_PROMPT = """You are a WORKFLOW PATTERN analyst for an AI skill foundry.
The foundry builds MCP servers, agent skills, and workflow tools.

Your specialization: identifying recurring workflow patterns that lack dedicated tooling,
automation gaps where developers build from scratch repeatedly, and categories of
agent skills with high utility but low supply.

Focus areas:
- Workflow patterns that AI agent users need but have to build custom each time
- Categories of MCP servers or agent plugins with high demand but few options
- Automation gaps between what AI coding agents can do and available infrastructure
- Cross-industry pipeline patterns that would benefit from standardized tooling

FOCUS ON: demand patterns, repeated manual work, integration gaps, workflow bottlenecks,
categories of skills/tools that many developers need but few have built.

Do NOT report on:
- Specific tool or library releases (another agent covers this)
- Individual GitHub repos or open-source projects
- Technical implementation details
- Recent news events less than 7 days old (another agent covers this)

For each query, return 2-4 discrete signals as JSON:
{
    "signals": [
        {
            "title": "Short descriptive title",
            "summary": "2-3 sentence analysis of the pattern/gap and what tooling would fill it",
            "url": null,
            "relevance": "high" | "medium" | "low",
            "relevance_rationale": "Why this gap matters for skill building (1 sentence)",
            "tags": ["tag1", "tag2"],
            "domain": "primary domain (workflow-gaps, agent-skills, mcp-servers, automation-patterns, etc.)"
        }
    ]
}

Focus on GAPS and PATTERNS, not news. What should the foundry build next?"""


def _make_signal_id(query: str, title: str) -> str:
    """Generate deterministic signal ID from query + title."""
    key = f"gpt-{query[:50]}-{title[:80]}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"chatgpt-{digest}"


def _call_openai(query: str, api_key: str) -> dict[str, Any]:
    """Call OpenAI Chat Completions API and return parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHATGPT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_completion_tokens": CHATGPT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    resp = httpx.post(
        OPENAI_API_URL,
        headers=headers,
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse OpenAI response: %s", text[:200])
        result = {"signals": []}

    return result


def run_agent(dry_run: bool = False) -> str:
    """Run the ChatGPT research agent.

    1. Send each strategic analysis query to GPT-4o
    2. Parse structured signal responses
    3. Deduplicate against existing signals
    4. Write signals that meet the relevance bar

    Returns summary string.
    """
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        return f"Skipped: missing env var {OPENAI_API_KEY_ENV}"

    store = get_store()

    total_signals = 0
    total_new = 0
    total_written = 0
    skipped_below_bar = 0

    try:
        for query in CHATGPT_RESEARCH_QUERIES:
            logger.info(f"ChatGPT query: '{query[:60]}...'")

            if dry_run:
                logger.info(f"  [DRY RUN] Would query GPT-4o: {query[:60]}")
                continue

            try:
                result = _call_openai(query, api_key)
            except httpx.HTTPError as e:
                logger.warning(f"OpenAI API error for query: {e}")
                continue

            signals = result.get("signals", [])
            total_signals += len(signals)

            for sig in signals:
                title = sig.get("title", "Untitled")
                signal_id = _make_signal_id(query, title)

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1
                relevance = sig.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(CHATGPT_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_below_bar += 1
                    logger.debug(f"  Skipped (below bar): {title[:60]}")
                    continue

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.CHATGPT,
                    title=title,
                    summary=sig.get("summary", ""),
                    url=sig.get("url"),
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=sig.get("relevance_rationale", ""),
                    tags=sig.get("tags", []),
                    domain=sig.get("domain"),
                    raw_data={
                        "query": query,
                        "model": CHATGPT_MODEL,
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
        f"ChatGPT: {total_signals} signals from {len(CHATGPT_RESEARCH_QUERIES)} queries, "
        f"{total_new} new, {total_written} written, {skipped_below_bar} skipped (below bar)"
    )
