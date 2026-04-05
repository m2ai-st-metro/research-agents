"""Tool/Library Monitor Agent.

Searches for new MCP servers, framework releases, and AI tooling updates.
Assesses relevance with persona-awareness tagging.

Assess-then-enrich: runs cheap metadata-based relevance first, then spends
a Firecrawl credit to scrape the repo README only for signals that pass.
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import (
    FIRECRAWL_ENRICH_MAX_CHARS,
    FIRECRAWL_ENRICH_MAX_PER_QUERY,
    PERSONA_IDS,
    TOOL_MAX_RESULTS_PER_QUERY,
    TOOL_SEARCH_QUERIES,
)
from ..firecrawl_client import is_enrichment_available, scrape_url
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

# GitHub API for searching repos (no auth needed for basic search)
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(source_key: str) -> str:
    """Generate a deterministic signal ID from a source identifier."""
    digest = hashlib.sha256(source_key.encode()).hexdigest()[:12]
    return f"tool-{digest}"


def _search_github_repos(query: str, max_results: int = 10) -> list[dict]:
    """Search GitHub for recently updated repos matching the query.

    Returns list of dicts with: full_name, description, url, stars, language, pushed_at
    """
    params = {
        "q": f"{query} pushed:>2026-02-01 stars:>=5",
        "sort": "updated",
        "order": "desc",
        "per_page": min(max_results, 30),
    }
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        resp = httpx.get(GITHUB_SEARCH_URL, params=params, headers=headers, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"GitHub API error for query '{query}': {e}")
        return []

    data = resp.json()
    repos = []
    for item in data.get("items", [])[:max_results]:
        repos.append({
            "full_name": item.get("full_name", ""),
            "description": (item.get("description") or "")[:300],
            "url": item.get("html_url", ""),
            "stars": item.get("stargazers_count", 0),
            "language": item.get("language"),
            "pushed_at": item.get("pushed_at", ""),
            "topics": item.get("topics", []),
        })

    return repos


def _enrich_summary(repo: dict, scraped_content: str) -> str:
    """Build an enriched summary from metadata + scraped README content."""
    meta = repo["description"] or repo["full_name"]
    excerpt = scraped_content[:2000].strip()
    if excerpt:
        return f"{meta}\n\nREADME excerpt:\n{excerpt}"
    return meta


def run_agent(dry_run: bool = False) -> str:
    """Run the tool/library monitor agent.

    1. Search GitHub for each configured query
    2. Deduplicate against existing signals
    3. Assess relevance via Claude API with persona-awareness (metadata first)
    4. Enrich with Firecrawl scrape for signals that pass relevance filter
    5. Write signals that pass the relevance threshold

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_low = 0
    total_enriched = 0

    try:
        for query in TOOL_SEARCH_QUERIES:
            logger.info(f"Searching GitHub: '{query}'")
            repos = _search_github_repos(query, max_results=TOOL_MAX_RESULTS_PER_QUERY)
            total_found += len(repos)
            enriched_this_query = 0

            for repo in repos:
                signal_id = _make_signal_id(repo["full_name"])

                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1

                if dry_run:
                    logger.info(f"  [DRY RUN] Would assess: {repo['full_name']}")
                    continue

                # First pass: cheap metadata-based relevance assessment
                assessment = assess_relevance(
                    title=repo["full_name"],
                    summary=repo["description"],
                    source_context=f"GitHub repo, {repo['stars']} stars, language: {repo['language']}, topics: {', '.join(repo.get('topics', [])[:5])}",
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                if RELEVANCE_ORDER.get(relevance, 0) < 2:  # Skip low
                    skipped_low += 1
                    logger.debug(f"  Skipped (low relevance): {repo['full_name']}")
                    continue

                # Enrichment: scrape repo page for README content
                summary = repo["description"] or repo["full_name"]
                enriched = False

                if (
                    enriched_this_query < FIRECRAWL_ENRICH_MAX_PER_QUERY
                    and is_enrichment_available()
                ):
                    scraped = scrape_url(
                        repo["url"], max_chars=FIRECRAWL_ENRICH_MAX_CHARS
                    )
                    if scraped:
                        summary = _enrich_summary(repo, scraped)
                        enriched_this_query += 1
                        total_enriched += 1
                        enriched = True
                        logger.info(f"  Enriched via Firecrawl: {repo['full_name']}")

                tags = assessment.get("tags", [])
                if enriched:
                    tags.append("firecrawl-enriched")

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.TOOL_MONITOR,
                    title=repo["full_name"],
                    summary=summary,
                    url=repo["url"],
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data={
                        "stars": repo["stars"],
                        "language": repo["language"],
                        "pushed_at": repo["pushed_at"],
                        "topics": repo.get("topics", []),
                        "firecrawl_enriched": enriched,
                    },
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {repo['full_name']}")

            # Rate limit courtesy
            time.sleep(2.0)

    finally:
        store.close()

    return (
        f"Found {total_found} repos, {total_new} new, "
        f"{total_written} written ({total_enriched} enriched), "
        f"{skipped_low} skipped (low relevance)"
    )
