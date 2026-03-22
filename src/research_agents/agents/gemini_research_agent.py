"""Gemini Research Agent.

Uses Gemini with Google Search grounding to discover signals with fresh web data.
Anti-monoculture: Google Search grounding surfaces different results than
Perplexity's index or GPT-4o's training data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from google import genai
from google.genai import types

from ..config import (
    GEMINI_API_KEY_ENV,
    GEMINI_RESEARCH_MAX_TOKENS,
    GEMINI_RESEARCH_MIN_RELEVANCE,
    GEMINI_RESEARCH_MODEL,
    GEMINI_RESEARCH_QUERIES,
)
from ..gemini_client import get_gemini_client
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}

SYSTEM_PROMPT = """You are an EMERGING TRENDS scout for a solo AI developer/consultant.
Your specialization: recent developments from the LAST 7 DAYS -- regulatory changes,
funding rounds, acquisitions, and breaking technical news. You report on what is NEW.

Focus areas:
- AI agent frameworks, MCP servers, tool-augmented LLMs
- Autonomous coding agents and AI-assisted software engineering
- Healthcare AI (HIPAA, home health, clinical tools)
- Developer productivity and workflow automation
- Small-team / solo-dev AI tooling

FOCUS ON: developments from the past 7 days, funding announcements, regulatory changes,
acquisitions, new company launches, breaking research results. Always include dates.

Do NOT report on:
- Established market dynamics or competitive analysis (another agent covers this)
- Tools or frameworks that have been available for more than 2 weeks
- Business model analysis or strategic positioning
- Broad industry trends without specific recent events

Use Google Search grounding to find the latest information. For each query,
return 2-5 discrete signals as JSON:
{
    "signals": [
        {
            "title": "Short descriptive title",
            "summary": "2-3 sentence summary with specific facts (dates, numbers, names)",
            "url": "Source URL if available from search results, or null",
            "relevance": "high" | "medium" | "low",
            "relevance_rationale": "Why this matters (1 sentence)",
            "tags": ["tag1", "tag2"],
            "domain": "primary domain (ai-agents, healthcare-ai, developer-tools, etc.)"
        }
    ]
}

Prioritize specificity and recency. Include dates, version numbers, company names."""


def _make_signal_id(query: str, title: str) -> str:
    """Generate deterministic signal ID from query + title."""
    key = f"gemini-{query[:50]}-{title[:80]}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"gemini-{digest}"


def _call_gemini(query: str, client: genai.Client) -> dict:
    """Call Gemini API with Google Search grounding and return parsed JSON."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nResearch query: {query}"

    # Enable Google Search grounding
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = client.models.generate_content(
        model=GEMINI_RESEARCH_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=GEMINI_RESEARCH_MAX_TOKENS,
            tools=[google_search_tool],
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip() if response.text else ""

    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini research response: %s", text[:200])
        result = {"signals": []}

    # Extract grounding metadata if available
    grounding_metadata = {}
    if response.candidates and response.candidates[0].grounding_metadata:
        gm = response.candidates[0].grounding_metadata
        if hasattr(gm, "search_entry_point") and gm.search_entry_point:
            grounding_metadata["search_query"] = getattr(
                gm.search_entry_point, "query", None
            )
        if hasattr(gm, "grounding_supports") and gm.grounding_supports:
            grounding_metadata["support_count"] = len(gm.grounding_supports)

    result["_grounding"] = grounding_metadata
    return result


def run_agent(dry_run: bool = False) -> str:
    """Run the Gemini research agent with Google Search grounding.

    1. Send each research query to Gemini with search grounding enabled
    2. Parse structured signal responses
    3. Deduplicate against existing signals
    4. Write signals that meet the relevance bar

    Returns summary string.
    """
    try:
        client = get_gemini_client()
    except RuntimeError as e:
        return f"Skipped: {e}"

    store = get_store()

    total_signals = 0
    total_new = 0
    total_written = 0
    skipped_below_bar = 0

    try:
        for query in GEMINI_RESEARCH_QUERIES:
            logger.info(f"Gemini research query: '{query[:60]}...'")

            if dry_run:
                logger.info(f"  [DRY RUN] Would query Gemini: {query[:60]}")
                continue

            try:
                result = _call_gemini(query, client)
            except Exception as e:
                logger.warning(f"Gemini API error for query: {e}")
                continue

            signals = result.get("signals", [])
            grounding = result.get("_grounding", {})
            total_signals += len(signals)

            for sig in signals:
                title = sig.get("title", "Untitled")
                signal_id = _make_signal_id(query, title)

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1
                relevance = sig.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(GEMINI_RESEARCH_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_below_bar += 1
                    logger.debug(f"  Skipped (below bar): {title[:60]}")
                    continue

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.GEMINI_RESEARCH,
                    title=title,
                    summary=sig.get("summary", ""),
                    url=sig.get("url"),
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=sig.get("relevance_rationale", ""),
                    tags=sig.get("tags", []),
                    domain=sig.get("domain"),
                    raw_data={
                        "query": query,
                        "model": GEMINI_RESEARCH_MODEL,
                        "grounding": grounding,
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
        f"Gemini: {total_signals} signals from {len(GEMINI_RESEARCH_QUERIES)} queries, "
        f"{total_new} new, {total_written} written, {skipped_below_bar} skipped (below bar)"
    )
