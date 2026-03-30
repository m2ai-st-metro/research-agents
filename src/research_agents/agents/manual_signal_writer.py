"""Manual signal ingestion utility.

Accepts a URL or topic description from Matthew's manual browsing,
fetches content if a URL is provided, runs relevance assessment,
and writes to ContractStore with source=manual.

Called by:
  - research-agents ingest <url_or_topic>  (CLI)
  - EA-Claude Telegram /signal handler     (via subprocess)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import FIRECRAWL_API_KEY_ENV
from ..signal_writer import signal_exists, write_signal

# Must come after signal_writer (which injects st-records into sys.path)
from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(url_or_topic: str) -> str:
    """Deterministic ID: manual-{sha256(input)[:12]}."""
    digest = hashlib.sha256(url_or_topic.encode()).hexdigest()[:12]
    return f"manual-{digest}"


def _is_url(text: str) -> bool:
    """Check if text looks like a URL."""
    return text.startswith("http://") or text.startswith("https://")


def _fetch_via_firecrawl(url: str, api_key: str) -> tuple[str, str]:
    """Use Firecrawl /scrape to get markdown content, extract title + summary."""
    try:
        resp = httpx.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url, "formats": ["markdown"]},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        markdown = data.get("data", {}).get("markdown", "")
        lines = [line.strip() for line in markdown.split("\n") if line.strip()]
        title = lines[0].lstrip("#").strip() if lines else url
        body_lines = [line for line in lines[1:] if not line.startswith("#")][:5]
        summary = " ".join(body_lines)[:500]
        return title, summary
    except Exception as e:
        logger.warning(f"Firecrawl fetch failed for {url}: {e}")
        return _fetch_via_httpx(url)


def _fetch_via_httpx(url: str) -> tuple[str, str]:
    """Basic httpx fetch, extract <title> tag and body text."""
    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "research-agents/1.0"},
        )
        resp.raise_for_status()
        text = resp.text

        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url

        body = re.sub(r"<[^>]+>", " ", text)
        body = re.sub(r"\s+", " ", body).strip()
        summary = body[:500]
        return title, summary
    except Exception as e:
        logger.warning(f"httpx fetch failed for {url}: {e}")
        return url, ""


def _fetch_url_content(url: str) -> tuple[str, str]:
    """Fetch title and summary from a URL.

    Prefers Firecrawl (structured extraction), falls back to httpx.
    Returns (title, summary).
    """
    api_key = os.environ.get(FIRECRAWL_API_KEY_ENV)
    if api_key:
        return _fetch_via_firecrawl(url, api_key)
    return _fetch_via_httpx(url)


def ingest_signal(url_or_topic: str, dry_run: bool = False) -> str:
    """Core ingestion logic. Called by both CLI and Telegram handler.

    Returns a human-readable result string suitable for both CLI output
    and Telegram reply.
    """
    signal_id = _make_signal_id(url_or_topic)

    if signal_exists(signal_id):
        return f"Already in signal store: {signal_id}"

    if dry_run:
        return f"[DRY RUN] Would ingest: {url_or_topic[:80]}"

    # Determine title + summary
    if _is_url(url_or_topic):
        url: str | None = url_or_topic
        logger.info(f"Fetching URL content: {url}")
        title, summary = _fetch_url_content(url_or_topic)
        source_context = "Manually shared URL by Matthew"
    else:
        url = None
        title = url_or_topic
        summary = url_or_topic
        source_context = "Manually shared topic by Matthew"

    # Run relevance assessment (for tagging/domain, not gating)
    client = get_client()
    assessment = assess_relevance(
        title=title,
        summary=summary,
        source_context=source_context,
        client=client,
    )

    relevance = assessment.get("relevance", "medium")
    # Manual signals: always write regardless of relevance
    # (Matthew curated it -- we still record the assessed level)

    tags: list[str] = assessment.get("tags", [])
    tags.append("manual")
    for persona in assessment.get("persona_tags", []):
        tags.append(f"persona:{persona}")

    write_signal(
        signal_id=signal_id,
        source=SignalSource.MANUAL,
        title=title,
        summary=summary,
        url=url,
        relevance=SignalRelevance(relevance),
        relevance_rationale=assessment.get("relevance_rationale", ""),
        tags=tags,
        domain=assessment.get("domain"),
        raw_data={"original_input": url_or_topic},
    )

    return (
        f"Signal saved.\n"
        f"ID: {signal_id}\n"
        f"Title: {title[:80]}\n"
        f"Relevance: {relevance}\n"
        f"Tags: {', '.join(tags[:6])}"
    )
