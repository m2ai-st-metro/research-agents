"""Adjacent Domain Watcher Agent.

Monitors healthcare AI, solo dev tools, and workflow automation trends.
Uses a higher relevance bar (HIGH only) since these are adjacent domains.

Assess-then-enrich: runs cheap metadata-based relevance first, then spends
a Firecrawl credit to scrape the full article only for signals that pass.
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import (
    DOMAIN_MIN_RELEVANCE,
    DOMAIN_WATCH_QUERIES,
    FIRECRAWL_ENRICH_MAX_CHARS,
    FIRECRAWL_ENRICH_MAX_PER_QUERY,
)
from ..firecrawl_client import is_enrichment_available, scrape_url
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

# Use Hacker News Algolia API for domain trend monitoring
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(source_key: str) -> str:
    """Generate a deterministic signal ID from a source identifier."""
    digest = hashlib.sha256(source_key.encode()).hexdigest()[:12]
    return f"domain-{digest}"


def _search_hn(query: str, max_results: int = 10) -> list[dict]:
    """Search Hacker News for recent stories matching the query.

    Returns list of dicts with: objectID, title, url, points, num_comments, created_at
    """
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": min(max_results, 20),
    }

    try:
        resp = httpx.get(HN_SEARCH_URL, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"HN API error for query '{query}': {e}")
        return []

    data = resp.json()
    stories = []
    for hit in data.get("hits", [])[:max_results]:
        stories.append({
            "objectID": hit.get("objectID", ""),
            "title": hit.get("title", ""),
            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            "points": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "created_at": hit.get("created_at", ""),
            "author": hit.get("author", ""),
        })

    return stories


def _enrich_summary(story: dict, scraped_content: str) -> str:
    """Build an enriched summary from metadata + scraped content."""
    meta = f"{story['points']} points, {story['num_comments']} comments on HN"
    # Take first ~2000 chars of scraped content as context
    excerpt = scraped_content[:2000].strip()
    if excerpt:
        return f"{meta}\n\nArticle excerpt:\n{excerpt}"
    return meta


def run_agent(dry_run: bool = False) -> str:
    """Run the adjacent domain watcher agent.

    1. Search HN for each configured domain query
    2. Deduplicate against existing signals
    3. Assess relevance via Claude API (metadata-only first pass)
    4. Enrich with Firecrawl scrape for signals that pass relevance bar
    5. Only write HIGH relevance signals (higher bar for adjacent domains)

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_below_bar = 0
    total_enriched = 0

    try:
        for query in DOMAIN_WATCH_QUERIES:
            logger.info(f"Searching HN for domain: '{query}'")
            stories = _search_hn(query, max_results=10)
            total_found += len(stories)
            enriched_this_query = 0

            for story in stories:
                signal_id = _make_signal_id(f"hn-{story['objectID']}")

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1

                if dry_run:
                    logger.info(f"  [DRY RUN] Would assess: {story['title'][:80]}")
                    continue

                # First pass: cheap metadata-based relevance assessment
                assessment = assess_relevance(
                    title=story["title"],
                    summary=f"HN story with {story['points']} points, {story['num_comments']} comments",
                    source_context=f"Hacker News story (domain watch: {query})",
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(DOMAIN_MIN_RELEVANCE, 3)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_below_bar += 1
                    logger.debug(f"  Skipped (below bar): {story['title'][:60]}")
                    continue

                # Enrichment: scrape full article for signals that passed
                summary = f"{story['points']} points, {story['num_comments']} comments on HN"
                enriched = False

                if (
                    enriched_this_query < FIRECRAWL_ENRICH_MAX_PER_QUERY
                    and is_enrichment_available()
                    and not story["url"].startswith("https://news.ycombinator.com")
                ):
                    scraped = scrape_url(
                        story["url"], max_chars=FIRECRAWL_ENRICH_MAX_CHARS
                    )
                    if scraped:
                        summary = _enrich_summary(story, scraped)
                        enriched_this_query += 1
                        total_enriched += 1
                        enriched = True
                        logger.info(f"  Enriched via Firecrawl: {story['title'][:60]}")

                tags = assessment.get("tags", [])
                if enriched:
                    tags.append("firecrawl-enriched")

                # Extract domain from the query category
                domain = assessment.get("domain")
                if not domain:
                    if "healthcare" in query.lower() or "hipaa" in query.lower() or "clinical" in query.lower():
                        domain = "healthcare-ai"
                    elif "solo" in query.lower() or "developer" in query.lower():
                        domain = "solo-dev-tools"
                    elif "workflow" in query.lower() or "automation" in query.lower():
                        domain = "workflow-automation"

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.DOMAIN_WATCH,
                    title=story["title"],
                    summary=summary,
                    url=story["url"],
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=domain,
                    raw_data={
                        "hn_id": story["objectID"],
                        "points": story["points"],
                        "num_comments": story["num_comments"],
                        "author": story["author"],
                        "created_at": story["created_at"],
                        "firecrawl_enriched": enriched,
                    },
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {story['title'][:60]}")

            # Rate limit courtesy
            time.sleep(1.0)

    finally:
        store.close()

    return (
        f"Found {total_found} stories, {total_new} new, "
        f"{total_written} written ({total_enriched} enriched), "
        f"{skipped_below_bar} skipped (below relevance bar)"
    )
