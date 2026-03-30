"""Reddit Scanner Agent.

Scans public subreddits for signals using Reddit's public JSON API
(no auth required). Targets communities where solo developers and
indie hackers discuss pain points and tools.

Signal IDs prefixed with `reddit-` for A2 source tracking.
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import (
    REDDIT_MAX_SIGNALS_PER_RUN,
    REDDIT_MIN_RELEVANCE,
    REDDIT_POSTS_PER_SUBREDDIT,
    REDDIT_SUBREDDITS,
)
from ..signal_writer import get_store, signal_exists, write_signal  # noqa: E402 — must come before contracts (injects sys.path)

from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402
from ..signal_writer import get_store, signal_exists, write_signal

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}

# Reddit public JSON API requires a User-Agent header
USER_AGENT = "research-agents/1.0 (ST Metro signal pipeline)"


def _make_signal_id(subreddit: str, post_id: str) -> str:
    """Generate a deterministic signal ID from subreddit + post ID."""
    key = f"reddit-{subreddit}-{post_id}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"reddit-{digest}"


def _fetch_subreddit_hot(subreddit: str, limit: int = 10) -> list[dict]:
    """Fetch hot posts from a subreddit via public JSON API.

    Returns list of dicts with: post_id, title, selftext, url, score, num_comments, subreddit
    """
    api_url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    headers = {"User-Agent": USER_AGENT}
    params = {"limit": limit, "raw_json": 1}

    try:
        resp = httpx.get(api_url, headers=headers, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Reddit API error for r/%s: %s", subreddit, e)
        return []

    data = resp.json()
    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        # Skip stickied/pinned posts
        if post.get("stickied"):
            continue
        posts.append({
            "post_id": post.get("id", ""),
            "title": post.get("title", ""),
            "selftext": (post.get("selftext") or "")[:500],
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "subreddit": subreddit,
        })

    return posts


def run_agent(dry_run: bool = False) -> str:
    """Run the Reddit scanner agent.

    1. Fetch hot posts from each configured subreddit
    2. Deduplicate against existing signals
    3. Assess relevance via Ollama
    4. Write signals that pass the relevance threshold
    5. Cap at REDDIT_MAX_SIGNALS_PER_RUN per run

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
        for subreddit in REDDIT_SUBREDDITS:
            if total_written >= REDDIT_MAX_SIGNALS_PER_RUN:
                break

            logger.info("Scanning r/%s", subreddit)
            posts = _fetch_subreddit_hot(subreddit, limit=REDDIT_POSTS_PER_SUBREDDIT)
            total_found += len(posts)

            for post in posts:
                if total_written >= REDDIT_MAX_SIGNALS_PER_RUN:
                    break

                signal_id = _make_signal_id(post["subreddit"], post["post_id"])

                if signal_exists(signal_id, store=store):
                    skipped_dedup += 1
                    continue

                total_new += 1

                if dry_run:
                    logger.info(
                        "  [DRY RUN] Would assess: r/%s - %s (%d pts)",
                        post["subreddit"], post["title"][:60], post["score"],
                    )
                    continue

                # Build context for relevance assessment
                summary = post["selftext"] if post["selftext"] else post["title"]
                assessment = assess_relevance(
                    title=post["title"],
                    summary=summary,
                    source_context=(
                        f"Reddit post from r/{post['subreddit']}, "
                        f"{post['score']} upvotes, {post['num_comments']} comments"
                    ),
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(REDDIT_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_low += 1
                    logger.debug("  Skipped (low relevance): %s", post["title"][:60])
                    continue

                tags = assessment.get("tags", [])
                tags.append(f"subreddit:{post['subreddit']}")
                for persona in assessment.get("persona_tags", []):
                    tags.append(f"persona:{persona}")

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.REDDIT,
                    title=post["title"],
                    summary=summary[:300],
                    url=post["url"],
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data={
                        "subreddit": post["subreddit"],
                        "score": post["score"],
                        "num_comments": post["num_comments"],
                        "post_id": post["post_id"],
                    },
                    store=store,
                )
                total_written += 1
                logger.info(
                    "  Wrote [%s]: r/%s - %s",
                    relevance, post["subreddit"], post["title"][:60],
                )

            # Rate limit between subreddits (Reddit is strict)
            time.sleep(3.0)

    finally:
        store.close()

    return (
        f"Reddit: {total_found} posts from {len(REDDIT_SUBREDDITS)} subreddits, "
        f"{total_new} new, {total_written} written, {skipped_low} skipped (low relevance), "
        f"{skipped_dedup} skipped (dedup)"
    )
