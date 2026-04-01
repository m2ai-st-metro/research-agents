"""ArXiv/Paper Scanner Agent.

Searches arXiv for papers relevant to the Snow-Town ecosystem,
assesses relevance via Claude API, and writes ResearchSignal records.
"""

from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import ARXIV_MAX_RESULTS_PER_QUERY, ARXIV_MIN_RELEVANCE, ARXIV_SEARCH_QUERIES
from ..signal_writer import get_store, signal_exists, write_signal

# Re-import after sys.path setup in signal_writer
from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _make_signal_id(arxiv_id: str) -> str:
    """Generate a deterministic signal ID from an arXiv paper ID."""
    return f"arxiv-{arxiv_id.replace('/', '-')}"


def _search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv API and return parsed paper metadata.

    Returns list of dicts with: arxiv_id, title, summary, url, authors, published
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            resp = httpx.get(ARXIV_API_URL, params=params, timeout=30.0, follow_redirects=True)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.info(f"arXiv rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except httpx.HTTPError as e:
            logger.warning(f"arXiv API error for query '{query}': {e}")
            return []
    else:
        logger.warning(f"arXiv rate limit persisted after 3 retries for query '{query}'")
        return []

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id_raw = entry.findtext("atom:id", "", ARXIV_NS)
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw

        title = entry.findtext("atom:title", "", ARXIV_NS).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", ARXIV_NS).strip().replace("\n", " ")
        published = entry.findtext("atom:published", "", ARXIV_NS)

        authors = []
        for author in entry.findall("atom:author", ARXIV_NS):
            name = author.findtext("atom:name", "", ARXIV_NS)
            if name:
                authors.append(name)

        # Get the PDF link
        url = arxiv_id_raw
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf":
                url = link.get("href", url)
                break

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary[:500],  # Truncate long abstracts
            "url": url,
            "authors": authors[:5],
            "published": published,
        })

    return papers


def run_agent(dry_run: bool = False) -> str:
    """Run the arXiv scanner agent.

    1. Search arXiv for each configured query
    2. Deduplicate against existing signals
    3. Assess relevance via Claude API
    4. Write signals that pass the relevance threshold

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_low = 0

    try:
        for query in ARXIV_SEARCH_QUERIES:
            logger.info(f"Searching arXiv: '{query}'")
            papers = _search_arxiv(query, max_results=ARXIV_MAX_RESULTS_PER_QUERY)
            total_found += len(papers)

            for paper in papers:
                signal_id = _make_signal_id(paper["arxiv_id"])

                # Dedup check
                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1

                if dry_run:
                    logger.info(f"  [DRY RUN] Would assess: {paper['title'][:80]}")
                    continue

                # Claude relevance assessment
                assessment = assess_relevance(
                    title=paper["title"],
                    summary=paper["summary"],
                    source_context=f"arXiv paper ({paper['arxiv_id']}), authors: {', '.join(paper['authors'][:3])}",
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                if RELEVANCE_ORDER.get(relevance, 0) < RELEVANCE_ORDER.get(ARXIV_MIN_RELEVANCE, 2):
                    skipped_low += 1
                    logger.debug(f"  Skipped (low relevance): {paper['title'][:60]}")
                    continue

                tags = assessment.get("tags", [])

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.ARXIV_HF,
                    title=paper["title"],
                    summary=paper["summary"],
                    url=paper["url"],
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data={
                        "arxiv_id": paper["arxiv_id"],
                        "authors": paper["authors"],
                        "published": paper["published"],
                    },
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {paper['title'][:60]}")

            # arXiv requires >= 3s between requests
            time.sleep(3.0)

    finally:
        store.close()

    return (
        f"Found {total_found} papers, {total_new} new, "
        f"{total_written} written, {skipped_low} skipped (low relevance)"
    )
