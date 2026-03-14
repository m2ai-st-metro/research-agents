"""Perplexity Web-Grounded Research Agent.

Uses the Perplexity Sonar API for web-grounded search across configured
research queries. Writes signals as JSON files to data/signals/perplexity/
and optionally to the Snow-Town ContractStore when available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import (
    DATA_DIR,
    PERPLEXITY_API_URL,
    PERPLEXITY_MAX_RESULTS_PER_QUERY,
    PERPLEXITY_MIN_RELEVANCE,
    PERPLEXITY_MODEL,
    PERPLEXITY_SEARCH_QUERIES,
)

logger = logging.getLogger(__name__)

SIGNALS_DIR = DATA_DIR / "signals" / "perplexity"
RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(query: str, index: int) -> str:
    """Generate a deterministic signal ID from query + index."""
    digest = hashlib.sha256(f"{query}:{index}".encode()).hexdigest()[:12]
    return f"pplx-{digest}"


def _get_perplexity_client() -> httpx.Client | None:
    """Create an httpx client configured for the Perplexity API.

    Returns None if PERPLEXITY_API_KEY is not set.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.warning("PERPLEXITY_API_KEY not set — running in offline mode")
        return None

    return httpx.Client(
        base_url="https://api.perplexity.ai",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )


def _query_perplexity(client: httpx.Client, query: str) -> list[dict]:
    """Query Perplexity Sonar API for web-grounded research results.

    Returns a list of signal dicts extracted from the response.
    """
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research assistant for a solo AI developer. "
                    "Return structured findings as a JSON array of objects with keys: "
                    "title, summary, url, relevance (high/medium/low), tags (list of strings). "
                    "Focus on actionable intelligence about tools, frameworks, papers, and trends."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research the following topic and return the top {PERPLEXITY_MAX_RESULTS_PER_QUERY} "
                    f"most relevant recent findings as JSON:\n\n{query}"
                ),
            },
        ],
    }

    for attempt in range(3):
        try:
            resp = client.post("/chat/completions", json=payload)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.info(f"Perplexity rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except httpx.HTTPError as e:
            logger.warning(f"Perplexity API error for query '{query}': {e}")
            return []
    else:
        logger.warning(f"Perplexity rate limit persisted after 3 retries for query '{query}'")
        return []

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = data.get("citations", [])

    # Parse the JSON array from the response
    results = _parse_results(content, query, citations)
    return results


def _parse_results(content: str, query: str, citations: list[str] | None = None) -> list[dict]:
    """Parse Perplexity response content into signal dicts."""
    # Try to extract JSON from the response
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(text)
        if isinstance(items, dict):
            items = [items]
    except json.JSONDecodeError:
        logger.warning(f"Could not parse Perplexity response as JSON for query '{query}'")
        # Fall back to treating the whole response as a single finding
        items = [
            {
                "title": f"Web research: {query}",
                "summary": text[:500],
                "url": None,
                "relevance": "medium",
                "tags": ["web-research"],
            }
        ]

    results = []
    for i, item in enumerate(items[:PERPLEXITY_MAX_RESULTS_PER_QUERY]):
        result = {
            "title": item.get("title", f"Finding {i + 1}: {query}"),
            "summary": item.get("summary", "")[:500],
            "url": item.get("url"),
            "relevance": item.get("relevance", "medium"),
            "tags": item.get("tags", []),
            "query": query,
        }
        # Attach citation URLs if available
        if citations:
            result["citations"] = citations
        results.append(result)

    return results


def _write_signal_file(signal_id: str, signal_data: dict) -> Path:
    """Write a signal as a JSON file to the local data directory."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SIGNALS_DIR / f"{signal_id}.json"
    filepath.write_text(json.dumps(signal_data, indent=2, default=str) + "\n")
    logger.info(f"Wrote signal file: {filepath.name}")
    return filepath


def _try_write_to_store(signal_id: str, signal_data: dict) -> bool:
    """Attempt to write signal to Snow-Town ContractStore. Returns False if unavailable."""
    try:
        from ..signal_writer import get_store, signal_exists, write_signal

        from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

        store = get_store()
        try:
            if signal_exists(signal_id, store=store):
                return True  # Already exists, skip

            write_signal(
                signal_id=signal_id,
                source=SignalSource.DOMAIN_WATCH,  # Closest available source type
                title=signal_data["title"],
                summary=signal_data["summary"],
                url=signal_data.get("url"),
                relevance=SignalRelevance(signal_data.get("relevance", "medium")),
                relevance_rationale=signal_data.get("relevance_rationale", "Perplexity web research"),
                tags=signal_data.get("tags", []) + ["perplexity"],
                domain=signal_data.get("domain"),
                raw_data=signal_data.get("raw_data"),
                store=store,
            )
            return True
        finally:
            store.close()
    except Exception as e:
        logger.debug(f"ContractStore unavailable, using local files only: {e}")
        return False


def run_agent(dry_run: bool = False) -> str:
    """Run the Perplexity web-grounded research agent.

    1. Query Perplexity Sonar API for each configured search query
    2. Parse and filter results by relevance
    3. Write signals as local JSON files (and to ContractStore if available)

    Returns summary string.
    """
    client = None if dry_run else _get_perplexity_client()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    total_queries = 0
    total_results = 0
    total_written = 0
    skipped_low = 0
    files_written: list[str] = []

    for query in PERPLEXITY_SEARCH_QUERIES:
        total_queries += 1
        logger.info(f"Researching: '{query}'")

        if dry_run or client is None:
            # Offline/dry-run mode: write query stub files for later processing
            for i in range(PERPLEXITY_MAX_RESULTS_PER_QUERY):
                signal_id = _make_signal_id(query, i)
                stub = {
                    "signal_id": signal_id,
                    "source": "perplexity",
                    "query": query,
                    "title": f"[pending] {query}",
                    "summary": f"Queued for Perplexity web research: {query}",
                    "relevance": "medium",
                    "tags": ["perplexity", "pending"],
                    "created_at": now.isoformat(),
                    "date": date_str,
                    "status": "dry_run" if dry_run else "offline",
                }
                filepath = _write_signal_file(signal_id, stub)
                files_written.append(filepath.name)
                total_results += 1
                total_written += 1

            logger.info(
                f"  [{'DRY RUN' if dry_run else 'OFFLINE'}] "
                f"Wrote {PERPLEXITY_MAX_RESULTS_PER_QUERY} stub signals for: {query[:60]}"
            )
            continue

        # Live mode: query Perplexity API
        results = _query_perplexity(client, query)
        total_results += len(results)

        for i, result in enumerate(results):
            signal_id = _make_signal_id(query, i)
            relevance = result.get("relevance", "medium")

            if RELEVANCE_ORDER.get(relevance, 0) < RELEVANCE_ORDER.get(PERPLEXITY_MIN_RELEVANCE, 2):
                skipped_low += 1
                logger.debug(f"  Skipped (low relevance): {result['title'][:60]}")
                continue

            signal_data = {
                "signal_id": signal_id,
                "source": "perplexity",
                "query": query,
                "title": result["title"],
                "summary": result["summary"],
                "url": result.get("url"),
                "relevance": relevance,
                "relevance_rationale": f"Perplexity web research for: {query}",
                "tags": result.get("tags", []) + ["perplexity"],
                "citations": result.get("citations", []),
                "created_at": now.isoformat(),
                "date": date_str,
                "status": "collected",
            }

            # Write local JSON file
            filepath = _write_signal_file(signal_id, signal_data)
            files_written.append(filepath.name)

            # Try ContractStore (best-effort)
            _try_write_to_store(signal_id, signal_data)

            total_written += 1
            logger.info(f"  Wrote [{relevance}]: {result['title'][:60]}")

        # Rate limit courtesy
        time.sleep(1.0)

    if client is not None:
        client.close()

    # Write a run summary file
    summary_data = {
        "run_date": now.isoformat(),
        "date": date_str,
        "queries": total_queries,
        "results": total_results,
        "written": total_written,
        "skipped_low": skipped_low,
        "files": files_written,
        "mode": "dry_run" if dry_run else ("offline" if client is None else "live"),
    }
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SIGNALS_DIR / f"run-summary-{date_str}.json"
    summary_path.write_text(json.dumps(summary_data, indent=2) + "\n")

    return (
        f"Queried {total_queries} topics, {total_results} results, "
        f"{total_written} written, {skipped_low} skipped (low relevance). "
        f"Files in {SIGNALS_DIR.relative_to(DATA_DIR.parent)}"
    )
