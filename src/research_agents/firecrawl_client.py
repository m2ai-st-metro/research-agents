"""Shared Firecrawl client for content enrichment.

Provides scrape + credit checking with a safety floor.
Used by domain_watcher and tool_monitor for assess-then-enrich pattern:
run cheap metadata-based relevance first, then spend a Firecrawl credit
only on signals worth enriching.

Credit usage is logged to data/firecrawl_usage.jsonl for test phase tracking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from .config import (
    DATA_DIR,
    FIRECRAWL_API_KEY_ENV,
    FIRECRAWL_CREDIT_FLOOR,
    FIRECRAWL_ENRICHMENT_ENABLED,
)

logger = logging.getLogger(__name__)

FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"
USAGE_LOG = DATA_DIR / "firecrawl_usage.jsonl"

# Module-level cache so we don't hammer the account endpoint every call.
_cached_credits: int | None = None
_credits_checked_at: float = 0.0
_CACHE_TTL = 300.0  # 5 minutes


def _get_api_key() -> str | None:
    return os.environ.get(FIRECRAWL_API_KEY_ENV)


def _log_usage(action: str, url: str, credits_before: int | None) -> None:
    """Append a line to the usage JSONL for post-test-phase review."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "action": action,
        "url": url[:200],
        "credits_before": credits_before,
    }
    try:
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning(f"Failed to write usage log: {e}")


def check_credits(force: bool = False) -> int:
    """Return remaining Firecrawl credits. Caches for 5 minutes.

    Returns -1 if the API key is missing or the request fails.
    """
    global _cached_credits, _credits_checked_at

    if not force and _cached_credits is not None:
        if (time.time() - _credits_checked_at) < _CACHE_TTL:
            return _cached_credits

    api_key = _get_api_key()
    if not api_key:
        logger.warning("FIRECRAWL_API_KEY not set")
        return -1

    try:
        resp = httpx.get(
            f"{FIRECRAWL_API_BASE}/account",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        credits = data.get("remaining_credits", data.get("credits", -1))
        _cached_credits = int(credits) if credits is not None else -1
        _credits_checked_at = time.time()
        logger.info(f"Firecrawl credits remaining: {_cached_credits}")
        return _cached_credits
    except Exception as e:
        logger.warning(f"Failed to check Firecrawl credits: {e}")
        return -1


def is_enrichment_available() -> bool:
    """Check if Firecrawl enrichment is enabled and has enough credits."""
    if not FIRECRAWL_ENRICHMENT_ENABLED:
        logger.debug("Firecrawl enrichment disabled via config")
        return False

    if not _get_api_key():
        logger.debug("FIRECRAWL_API_KEY not set")
        return False

    credits = check_credits()
    if credits == -1:
        return False

    if credits < FIRECRAWL_CREDIT_FLOOR:
        logger.warning(
            f"Firecrawl credits ({credits}) below floor ({FIRECRAWL_CREDIT_FLOOR}), "
            "skipping enrichment"
        )
        return False

    return True


def scrape_url(url: str, max_chars: int = 3000) -> str | None:
    """Scrape a URL via Firecrawl API. Returns markdown content or None on failure.

    Checks credit floor before scraping. Logs usage for tracking.
    Truncates output to max_chars to keep Claude API costs reasonable.
    """
    if not is_enrichment_available():
        return None

    api_key = _get_api_key()
    if not api_key:
        return None

    global _cached_credits

    credits_before = _cached_credits
    _log_usage("scrape", url, credits_before)

    try:
        resp = httpx.post(
            f"{FIRECRAWL_API_BASE}/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        markdown = data.get("data", {}).get("markdown", "")

        # Invalidate credit cache so next check gets fresh count.
        _cached_credits = None

        if not markdown:
            logger.warning(f"Empty markdown from Firecrawl for {url}")
            return None

        return markdown[:max_chars]

    except Exception as e:
        logger.warning(f"Firecrawl scrape failed for {url}: {e}")
        return None
