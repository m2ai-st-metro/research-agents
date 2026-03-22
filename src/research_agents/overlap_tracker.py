"""Pairwise signal overlap measurement for anti-monoculture monitoring.

Computes Jaccard similarity on title tokens between agents' signal sets
to detect when agents are producing redundant signals. Generates weekly
reports to data/overlap_reports/.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

from .config import DATA_DIR, SNOW_TOWN_ROOT

logger = logging.getLogger(__name__)

# Ensure st-factory contracts importable
_snow_town_path = str(SNOW_TOWN_ROOT)
if _snow_town_path not in sys.path:
    sys.path.insert(0, _snow_town_path)

from contracts.store import ContractStore  # noqa: E402

OVERLAP_REPORT_DIR = DATA_DIR / "overlap_reports"

# Stop words to exclude from token comparison
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "that", "this", "are", "was",
    "be", "has", "had", "not", "no", "as", "its", "new", "ai", "tool",
})


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words, filtering stop words and short tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compute_pairwise_overlap(
    days: int = 7,
    data_dir: Path | None = None,
) -> dict[str, float]:
    """Compute pairwise Jaccard overlap between signal sources.

    Args:
        days: Lookback window in days (default 7 for weekly).
        data_dir: Override data dir for ContractStore (tests).

    Returns:
        Dict mapping "source_a vs source_b" to Jaccard similarity score.
    """
    store_dir = data_dir or (SNOW_TOWN_ROOT / "data")
    store = ContractStore(data_dir=store_dir)

    try:
        all_signals = store.query_signals(limit=5000)
    finally:
        store.close()

    cutoff = datetime.now() - timedelta(days=days)

    # Group title tokens by source
    source_tokens: dict[str, set[str]] = {}
    for s in all_signals:
        if s.emitted_at < cutoff:
            continue
        source = s.source.value
        if source not in source_tokens:
            source_tokens[source] = set()
        source_tokens[source] |= _tokenize(s.title)
        source_tokens[source] |= _tokenize(s.summary[:100])

    # Compute pairwise
    overlaps: dict[str, float] = {}
    sources = sorted(source_tokens.keys())

    for src_a, src_b in combinations(sources, 2):
        score = _jaccard_similarity(source_tokens[src_a], source_tokens[src_b])
        key = f"{src_a} vs {src_b}"
        overlaps[key] = round(score, 3)

    return overlaps


def generate_overlap_report(days: int = 7) -> Path:
    """Generate a weekly overlap report and save to data/overlap_reports/.

    Returns path to the generated report.
    """
    overlaps = compute_pairwise_overlap(days=days)
    now = datetime.now(timezone.utc)

    report = {
        "generated_at": now.isoformat(),
        "window_days": days,
        "pairwise_overlaps": overlaps,
        "high_overlap_pairs": {
            k: v for k, v in overlaps.items() if v > 0.3
        },
        "summary": {
            "total_pairs": len(overlaps),
            "avg_overlap": round(
                sum(overlaps.values()) / len(overlaps), 3
            ) if overlaps else 0.0,
            "max_overlap": max(overlaps.values()) if overlaps else 0.0,
            "max_overlap_pair": (
                max(overlaps, key=overlaps.get)  # type: ignore[arg-type]
                if overlaps else "none"
            ),
        },
    }

    OVERLAP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"overlap_{now.strftime('%Y%m%d')}.json"
    report_path = OVERLAP_REPORT_DIR / filename

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Overlap report: avg=%.3f, max=%.3f (%s), saved to %s",
        report["summary"]["avg_overlap"],
        report["summary"]["max_overlap"],
        report["summary"]["max_overlap_pair"],
        report_path,
    )

    return report_path
