"""GitHub Trending Harvester Agent.

Searches GitHub for recently created/updated repositories with significant
star counts across key languages. Assesses relevance with persona-awareness
tagging, then writes signals to ST Factory ContractStore.

Signal IDs prefixed with `github_trending-` for A1 source tracking.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import PERSONA_IDS
from ..signal_writer import get_store, signal_exists, write_signal

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

# GitHub API endpoint
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

# Configuration
GITHUB_TRENDING_LANGUAGES: list[str] = ["python", "typescript", "javascript", "rust"]
GITHUB_TRENDING_MIN_STARS: int = 50
GITHUB_TRENDING_LOOKBACK_DAYS: int = 7
GITHUB_TRENDING_MAX_REPOS_PER_LANGUAGE: int = 10
GITHUB_TRENDING_MAX_SIGNALS_PER_RUN: int = 10

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(repo_full_name: str) -> str:
    """Generate a deterministic signal ID from a repo full name."""
    digest = hashlib.sha256(repo_full_name.encode()).hexdigest()[:12]
    return f"github_trending-{digest}"


def _get_headers() -> dict[str, str]:
    """Build GitHub API headers, using token if available."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _search_trending_repos(
    language: str,
    min_stars: int = GITHUB_TRENDING_MIN_STARS,
    lookback_days: int = GITHUB_TRENDING_LOOKBACK_DAYS,
    max_results: int = GITHUB_TRENDING_MAX_REPOS_PER_LANGUAGE,
) -> list[dict]:
    """Search GitHub for trending repos in a specific language.

    Returns list of dicts with: full_name, description, url, stars, language, pushed_at, topics
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = f"language:{language} stars:>={min_stars} pushed:>{cutoff}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(max_results, 30),
    }

    try:
        resp = httpx.get(
            GITHUB_SEARCH_URL,
            params=params,
            headers=_get_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("GitHub API error for language '%s': %s", language, e)
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
            "created_at": item.get("created_at", ""),
        })

    return repos


def run_agent(dry_run: bool = False) -> str:
    """Run the GitHub trending harvester agent.

    1. Search GitHub for each configured language, sorted by stars
    2. Deduplicate against existing signals (both github_trending and tool_monitor)
    3. Assess relevance via Claude API with persona-awareness
    4. Write signals that pass the relevance threshold
    5. Cap at GITHUB_TRENDING_MAX_SIGNALS_PER_RUN signals per run

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
        for language in GITHUB_TRENDING_LANGUAGES:
            if total_written >= GITHUB_TRENDING_MAX_SIGNALS_PER_RUN:
                break

            logger.info("Searching GitHub trending: language=%s", language)
            repos = _search_trending_repos(language)
            total_found += len(repos)

            for repo in repos:
                if total_written >= GITHUB_TRENDING_MAX_SIGNALS_PER_RUN:
                    break

                signal_id = _make_signal_id(repo["full_name"])

                # Dedup: check both github_trending and tool_monitor signals
                if signal_exists(signal_id, store=store):
                    skipped_dedup += 1
                    continue

                # Also check if tool_monitor already has this repo
                tool_monitor_id = f"tool-{hashlib.sha256(repo['full_name'].encode()).hexdigest()[:12]}"
                if signal_exists(tool_monitor_id, store=store):
                    skipped_dedup += 1
                    continue

                total_new += 1

                if dry_run:
                    logger.info(
                        "  [DRY RUN] Would assess: %s (%d stars, %s)",
                        repo["full_name"], repo["stars"], repo["language"],
                    )
                    continue

                # Assess relevance via Ollama (same as tool_monitor)
                assessment = assess_relevance(
                    title=repo["full_name"],
                    summary=repo["description"],
                    source_context=(
                        f"GitHub trending repo, {repo['stars']} stars, "
                        f"language: {repo['language']}, "
                        f"topics: {', '.join(repo.get('topics', [])[:5])}"
                    ),
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                if RELEVANCE_ORDER.get(relevance, 0) < 2:  # Skip low
                    skipped_low += 1
                    logger.debug("  Skipped (low relevance): %s", repo["full_name"])
                    continue

                tags = assessment.get("tags", [])
                for persona in assessment.get("persona_tags", []):
                    tags.append(f"persona:{persona}")

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.GITHUB_TRENDING,
                    title=repo["full_name"],
                    summary=repo["description"] or repo["full_name"],
                    url=repo["url"],
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data={
                        "stars": repo["stars"],
                        "language": repo["language"],
                        "pushed_at": repo["pushed_at"],
                        "created_at": repo["created_at"],
                        "topics": repo.get("topics", []),
                    },
                    store=store,
                )
                total_written += 1
                logger.info("  Wrote [%s]: %s (%d stars)", relevance, repo["full_name"], repo["stars"])

            # Rate limit courtesy
            time.sleep(2.0)

    finally:
        store.close()

    return (
        f"Found {total_found} repos, {total_new} new, "
        f"{total_written} written, {skipped_low} skipped (low relevance), "
        f"{skipped_dedup} skipped (dedup)"
    )
