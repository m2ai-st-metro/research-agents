"""RSS/Newsletter Scanner Agent.

Ingests AI newsletter RSS feeds, extracts articles, assesses relevance,
and writes ResearchSignal records to ContractStore.

Feeds configured in config.RSS_FEEDS. Uses feedparser for standard RSS/Atom,
Firecrawl API for sources without usable RSS (e.g. The Batch).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from calendar import timegm
from datetime import datetime, timedelta, timezone

import feedparser  # type: ignore[import-untyped]
import httpx

from ..claude_client import assess_relevance, get_client
from ..config import (
    FIRECRAWL_API_KEY_ENV,
    RSS_FEEDS,
    RSS_LOOKBACK_DAYS,
    RSS_MIN_RELEVANCE,
)
from ..signal_writer import get_store, signal_exists, write_signal

# Must come after signal_writer (which injects st-records into sys.path)
from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(url: str) -> str:
    """Generate a deterministic signal ID from an article URL."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"rss-{digest}"


def _fetch_via_feedparser(feed_config: dict[str, str]) -> list[dict[str, object]]:
    """Parse an RSS/Atom feed using feedparser.

    Returns list of dicts with: title, url, summary, published, feed_name.
    """
    result = feedparser.parse(feed_config["url"])
    articles: list[dict[str, object]] = []

    for entry in result.entries:
        published = entry.get("published_parsed")
        articles.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "summary": str(entry.get("summary", entry.get("description", "")))[:500],
            "published": published,
            "feed_name": feed_config["name"],
        })

    return articles


def _fetch_via_firecrawl(feed_config: dict[str, str]) -> list[dict[str, object]]:
    """Fetch a page via Firecrawl API and extract article links.

    Used for sources without usable RSS (e.g. The Batch).
    Returns same structure as _fetch_via_feedparser.
    """
    api_key = os.environ.get(FIRECRAWL_API_KEY_ENV)
    if not api_key:
        logger.warning(f"FIRECRAWL_API_KEY not set, skipping {feed_config['name']}")
        return []

    try:
        resp = httpx.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": feed_config["url"], "formats": ["markdown"]},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        markdown = data.get("data", {}).get("markdown", "")

        # Extract article links from markdown: [Title](url)
        link_pattern = re.compile(
            r'\[([^\]]+)\]\((https://www\.deeplearning\.ai/the-batch/[^\s)]+)\)'
        )
        matches = link_pattern.findall(markdown)

        articles: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        for title, url in matches[:10]:  # cap at 10 links
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append({
                "title": title.strip(),
                "url": url,
                "summary": "",  # Will be assessed by Claude
                "published": None,  # No date from scrape
                "feed_name": feed_config["name"],
            })

        logger.info(f"Firecrawl extracted {len(articles)} links from {feed_config['name']}")
        return articles[:5]  # Return at most 5

    except Exception as e:
        logger.warning(f"Firecrawl fetch failed for {feed_config['name']}: {e}")
        return []


def _is_within_lookback(entry: dict[str, object], days: int) -> bool:
    """True if the article was published within the lookback window.

    Articles with no parsed date are assumed recent (within lookback).
    """
    published = entry.get("published")
    if published is None:
        return True  # No date available, assume recent

    try:
        if hasattr(published, "tm_year"):
            # time.struct_time from feedparser
            ts = timegm(published)  # type: ignore[arg-type]
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            return True  # Unknown format, assume recent
    except (ValueError, OverflowError):
        return True

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    return pub_dt >= cutoff


def run_agent(dry_run: bool = False) -> str:
    """Run the RSS scanner agent.

    1. Iterate configured feeds (feedparser or firecrawl)
    2. Filter by lookback window
    3. Deduplicate against existing signals
    4. Assess relevance via Claude API
    5. Write signals that pass the relevance threshold

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_low = 0
    skipped_old = 0

    try:
        for feed_config in RSS_FEEDS:
            logger.info(f"Fetching feed: {feed_config['name']} ({feed_config['parser']})")

            if feed_config["parser"] == "feedparser":
                articles = _fetch_via_feedparser(feed_config)
            else:
                articles = _fetch_via_firecrawl(feed_config)

            total_found += len(articles)

            for article in articles:
                if not _is_within_lookback(article, RSS_LOOKBACK_DAYS):
                    skipped_old += 1
                    continue

                article_url = str(article.get("url", ""))
                if not article_url:
                    continue

                signal_id = _make_signal_id(article_url)

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1

                if dry_run:
                    title = str(article.get("title", ""))
                    logger.info(f"  [DRY RUN] Would assess: {title[:80]}")
                    continue

                article_title = str(article.get("title", ""))
                article_summary = str(article.get("summary", ""))
                feed_name = str(article.get("feed_name", ""))

                assessment = assess_relevance(
                    title=article_title,
                    summary=article_summary or article_title,
                    source_context=f"RSS newsletter: {feed_name}",
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                if RELEVANCE_ORDER.get(relevance, 0) < RELEVANCE_ORDER.get(
                    RSS_MIN_RELEVANCE, 2
                ):
                    skipped_low += 1
                    logger.debug(f"  Skipped (low relevance): {article_title[:60]}")
                    continue

                tags = assessment.get("tags", [])

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.RSS_SCANNER,
                    title=article_title,
                    summary=article_summary or article_title,
                    url=article_url,
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data={"feed_name": feed_name},
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {article_title[:60]}")

            time.sleep(1.0)

    finally:
        store.close()

    return (
        f"Found {total_found} articles, {total_new} new, "
        f"{total_written} written, {skipped_low} skipped (low relevance), "
        f"{skipped_old} skipped (too old)"
    )
