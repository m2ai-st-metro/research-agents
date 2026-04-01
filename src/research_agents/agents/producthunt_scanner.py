"""Product Hunt Scanner Agent.

Scans Product Hunt's RSS feed for new product launches using feedparser.
Assesses relevance via Ollama, writes signals to ContractStore.

Signal IDs prefixed with `producthunt-` for A2 source tracking.
"""

from __future__ import annotations

import hashlib
import logging
import time
from calendar import timegm
from datetime import datetime, timedelta, timezone

import feedparser  # type: ignore[import-untyped]

from ..claude_client import assess_relevance, get_client
from ..config import (
    PRODUCTHUNT_MAX_ITEMS,
    PRODUCTHUNT_MAX_SIGNALS_PER_RUN,
    PRODUCTHUNT_MIN_RELEVANCE,
    PRODUCTHUNT_RSS_URL,
)
from ..signal_writer import get_store, signal_exists, write_signal  # noqa: E402 — must come before contracts (injects sys.path)

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(url: str) -> str:
    """Generate a deterministic signal ID from a product URL."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"producthunt-{digest}"


def _fetch_producthunt_feed() -> list[dict]:
    """Parse Product Hunt RSS feed via feedparser.

    Returns list of dicts with: title, summary, url, published.
    """
    feed = feedparser.parse(PRODUCTHUNT_RSS_URL)

    if feed.bozo and not feed.entries:
        logger.warning("Product Hunt RSS parse error: %s", feed.bozo_exception)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    items: list[dict] = []

    for entry in feed.entries[:PRODUCTHUNT_MAX_ITEMS]:
        # Parse publication date
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            pub_dt = datetime.fromtimestamp(timegm(published_parsed), tz=timezone.utc)
            if pub_dt < cutoff:
                continue
        else:
            pub_dt = None

        # Extract summary/description
        summary = entry.get("summary", "") or entry.get("description", "")
        # Strip HTML tags (simple approach)
        import re
        summary = re.sub(r"<[^>]+>", "", summary)[:500]

        items.append({
            "title": entry.get("title", "Untitled"),
            "summary": summary,
            "url": entry.get("link", ""),
            "published": pub_dt.isoformat() if pub_dt else None,
        })

    return items


def run_agent(dry_run: bool = False) -> str:
    """Run the Product Hunt scanner agent.

    1. Fetch recent items from Product Hunt RSS feed
    2. Deduplicate against existing signals
    3. Assess relevance via Ollama
    4. Write signals that pass the relevance threshold
    5. Cap at PRODUCTHUNT_MAX_SIGNALS_PER_RUN per run

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_low = 0
    skipped_dedup = 0

    try:
        logger.info("Fetching Product Hunt RSS feed")
        items = _fetch_producthunt_feed()
        total_found = len(items)
        logger.info("Found %d recent Product Hunt items", total_found)

        for item in items:
            if total_written >= PRODUCTHUNT_MAX_SIGNALS_PER_RUN:
                break

            if not item["url"]:
                continue

            signal_id = _make_signal_id(item["url"])

            if signal_exists(signal_id, store=store):
                skipped_dedup += 1
                continue

            total_new += 1

            if dry_run:
                logger.info(
                    "  [DRY RUN] Would assess: %s",
                    item["title"][:60],
                )
                continue

            assessment = assess_relevance(
                title=item["title"],
                summary=item["summary"],
                source_context="Product Hunt launch — new product/tool listing",
                client=client,
            )

            relevance = assessment.get("relevance", "low")
            min_level = RELEVANCE_ORDER.get(PRODUCTHUNT_MIN_RELEVANCE, 2)
            if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                skipped_low += 1
                logger.debug("  Skipped (low relevance): %s", item["title"][:60])
                continue

            tags = assessment.get("tags", [])
            tags.append("source:producthunt")

            write_signal(
                signal_id=signal_id,
                source=SignalSource.PRODUCT_HUNT,
                title=item["title"],
                summary=item["summary"][:300],
                url=item["url"],
                relevance=SignalRelevance(relevance),
                relevance_rationale=assessment.get("relevance_rationale", ""),
                tags=tags,
                domain=assessment.get("domain"),
                raw_data={
                    "published": item["published"],
                    "feed_url": PRODUCTHUNT_RSS_URL,
                },
                store=store,
            )
            total_written += 1
            logger.info("  Wrote [%s]: %s", relevance, item["title"][:60])

            # Rate limit between assessments
            time.sleep(1.0)

    finally:
        store.close()

    return (
        f"Product Hunt: {total_found} items, "
        f"{total_new} new, {total_written} written, {skipped_low} skipped (low relevance), "
        f"{skipped_dedup} skipped (dedup)"
    )
