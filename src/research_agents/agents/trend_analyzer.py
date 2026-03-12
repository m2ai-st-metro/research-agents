"""Trend Analyzer Agent.

Reads accumulated signals from the past N days, clusters by domain and tags,
detects rising/falling themes, and synthesizes a trend report via Claude.

Outputs:
  1. Markdown file at data/trend_reports/YYYY-MM-DD.md
  2. A synthetic ResearchSignal (source=trend_analyzer) summarizing the report
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from ..claude_client import get_client
from ..config import (
    TREND_LOOKBACK_DAYS,
    TREND_MIN_SIGNALS_FOR_ANALYSIS,
    TREND_REPORT_DIR,
    TREND_SUMMARIZER_MAX_TOKENS,
    TREND_SUMMARIZER_MODEL,
)
from ..signal_writer import get_store, signal_exists, write_signal

# Must come after signal_writer (which injects st-factory into sys.path)
from contracts.research_signal import (  # noqa: E402
    ResearchSignal,
    SignalRelevance,
    SignalSource,
)

logger = logging.getLogger(__name__)


def _load_signals(days: int) -> list[ResearchSignal]:
    """Load all signals from the past N days, any relevance level."""
    store = get_store()
    try:
        all_signals = store.query_signals(limit=2000)
        cutoff = datetime.now() - timedelta(days=days)
        return [s for s in all_signals if s.emitted_at >= cutoff]
    finally:
        store.close()


def _cluster_signals(
    signals: list[ResearchSignal],
) -> dict[str, object]:
    """Group signals by domain and by tag frequency.

    Returns dict with:
        by_domain: {domain_str: [signal, ...]}
        tag_counts: Counter of tag -> count
        by_source: {source_str: count}
        top_tags: list of (tag, count) sorted descending
    """
    by_domain: dict[str, list[ResearchSignal]] = defaultdict(list)
    tag_counter: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for s in signals:
        domain_key = s.domain or "general"
        by_domain[domain_key].append(s)
        for tag in s.tags:
            if not tag.startswith("persona:"):
                tag_counter[tag] += 1
        by_source[s.source.value] += 1

    return {
        "by_domain": dict(by_domain),
        "tag_counts": tag_counter,
        "by_source": dict(by_source),
        "top_tags": tag_counter.most_common(20),
    }


def _detect_rising_themes(
    signals: list[ResearchSignal], window_days: int
) -> dict[str, list[tuple[str, int, int]]]:
    """Compare first half vs second half of the window to detect trends.

    Rising: more signals in second half than first half (min 2 in second).
    Falling: fewer signals in second half than first half (min 2 in first).
    """
    cutoff = datetime.now() - timedelta(days=window_days)
    midpoint = cutoff + timedelta(days=window_days // 2)

    first_half = [s for s in signals if s.emitted_at < midpoint]
    second_half = [s for s in signals if s.emitted_at >= midpoint]

    first_tags: Counter[str] = Counter()
    second_tags: Counter[str] = Counter()

    for s in first_half:
        for tag in s.tags:
            if not tag.startswith("persona:"):
                first_tags[tag] += 1
    for s in second_half:
        for tag in s.tags:
            if not tag.startswith("persona:"):
                second_tags[tag] += 1

    all_tags = set(first_tags.keys()) | set(second_tags.keys())

    rising: list[tuple[str, int, int]] = []
    falling: list[tuple[str, int, int]] = []

    for tag in all_tags:
        f = first_tags.get(tag, 0)
        sec = second_tags.get(tag, 0)
        if sec > f and sec >= 2:
            rising.append((tag, f, sec))
        elif f > sec and f >= 2:
            falling.append((tag, f, sec))

    rising.sort(key=lambda x: x[2] - x[1], reverse=True)
    falling.sort(key=lambda x: x[1] - x[2], reverse=True)

    return {"rising": rising[:10], "falling": falling[:5]}


def _build_domain_digest(
    by_domain: dict[str, list[ResearchSignal]],
) -> str:
    """Build a compact signal digest grouped by domain for the prompt."""
    parts: list[str] = []
    for domain, domain_signals in sorted(
        by_domain.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        top3 = sorted(
            domain_signals, key=lambda s: RELEVANCE_ORD.get(s.relevance.value, 0), reverse=True
        )[:3]
        parts.append(f"\n### {domain} ({len(domain_signals)} signals)")
        for s in top3:
            parts.append(f"- [{s.source.value}] {s.title}: {s.summary[:100]}")
    return "\n".join(parts)


RELEVANCE_ORD = {"high": 3, "medium": 2, "low": 1}


def _synthesize_trend_report(
    signals: list[ResearchSignal],
    clusters: dict[str, object],
    trends: dict[str, list[tuple[str, int, int]]],
    dry_run: bool = False,
) -> str:
    """Use Claude Haiku to synthesize a trend report.

    Returns markdown string.
    """
    if dry_run:
        return f"[DRY RUN] Would synthesize trend report from {len(signals)} signals"

    by_domain: dict[str, list[ResearchSignal]] = clusters.get("by_domain", {})  # type: ignore[assignment]
    domain_digest = _build_domain_digest(by_domain)

    prompt = f"""You are analyzing research signals for a solo AI developer/consultant. \
Write a weekly trend report.

Data summary:
- Total signals: {len(signals)} over {TREND_LOOKBACK_DAYS} days
- Sources: {clusters.get('by_source', {})}
- Top tags: {list(clusters.get('top_tags', []))[:15]}
- Rising themes (tag, first_half_count, second_half_count): {trends['rising'][:5]}
- Falling themes (tag, first_half_count, second_half_count): {trends['falling'][:3]}

Sample signals by domain:
{domain_digest}

Write a concise markdown trend report with these sections:
1. **Executive Summary** (2-3 sentences, most important takeaways)
2. **Rising Themes** (what's gaining traction, cite specific signals as evidence)
3. **Falling Themes** (what's cooling off)
4. **Domain Spotlight** (most active domain this week, why it matters)
5. **Recommended Actions** (1-3 concrete things to pay attention to or prototype)

Keep it punchy. This is for a practitioner who ships, not a researcher who reads. No fluff."""

    client = get_client()
    response = client.messages.create(
        model=TREND_SUMMARIZER_MODEL,
        max_tokens=TREND_SUMMARIZER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()  # type: ignore[union-attr]


def _write_markdown_report(content: str, date_str: str) -> Path:
    """Write trend report to data/trend_reports/YYYY-MM-DD.md."""
    TREND_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = TREND_REPORT_DIR / f"{date_str}.md"
    path.write_text(content)
    return path


def run_agent(dry_run: bool = False) -> str:
    """Run the trend analyzer.

    1. Load signals from the past TREND_LOOKBACK_DAYS
    2. Cluster by domain and tag frequency
    3. Detect rising/falling themes (first half vs second half)
    4. Synthesize markdown report via Claude Haiku
    5. Write report to data/trend_reports/YYYY-MM-DD.md
    6. Write synthetic signal to ContractStore

    Returns summary string.
    """
    signals = _load_signals(TREND_LOOKBACK_DAYS)
    logger.info(f"Loaded {len(signals)} signals for trend analysis")

    if len(signals) < TREND_MIN_SIGNALS_FOR_ANALYSIS:
        return (
            f"Insufficient signals ({len(signals)} < "
            f"{TREND_MIN_SIGNALS_FOR_ANALYSIS}). Skipping."
        )

    clusters = _cluster_signals(signals)
    trends = _detect_rising_themes(signals, TREND_LOOKBACK_DAYS)

    date_str = datetime.now().strftime("%Y-%m-%d")

    if dry_run:
        by_domain: dict[str, list[ResearchSignal]] = clusters.get("by_domain", {})  # type: ignore[assignment]
        logger.info(f"[DRY RUN] Clusters: {list(by_domain.keys())}")
        logger.info(f"[DRY RUN] Rising: {trends['rising'][:3]}")
        return f"[DRY RUN] {len(signals)} signals, {len(by_domain)} domains"

    report_md = _synthesize_trend_report(signals, clusters, trends)
    report_path = _write_markdown_report(report_md, date_str)
    logger.info(f"Wrote trend report: {report_path}")

    # Extract executive summary for the signal record
    summary_lines = [
        line for line in report_md.split("\n") if line.strip() and not line.startswith("#")
    ]
    summary = " ".join(summary_lines[:3])[:500]

    signal_id = f"trend-{date_str}"

    # Skip if already ran today
    if signal_exists(signal_id):
        logger.info(f"Trend signal {signal_id} already exists, skipping write")
    else:
        top_tags_list: list[tuple[str, int]] = clusters.get("top_tags", [])  # type: ignore[assignment]
        by_domain_map: dict[str, list[ResearchSignal]] = clusters.get("by_domain", {})  # type: ignore[assignment]

        write_signal(
            signal_id=signal_id,
            source=SignalSource.TREND_ANALYZER,
            title=f"Weekly Trend Report: {date_str}",
            summary=summary,
            url=None,
            relevance=SignalRelevance.HIGH,
            relevance_rationale="Auto-generated weekly trend synthesis",
            tags=["trend-report", "weekly-digest"]
            + [t for t, _ in top_tags_list[:5]],
            domain="trend-analysis",
            raw_data={
                "signal_count": len(signals),
                "domains": list(by_domain_map.keys()),
                "rising_themes": [t for t, _, _ in trends["rising"][:5]],
                "report_path": str(report_path),
            },
        )

    return (
        f"Analyzed {len(signals)} signals, "
        f"{len(clusters.get('by_domain', {}))} domains, "  # type: ignore[arg-type]
        f"{len(trends['rising'])} rising themes. "
        f"Report: {report_path}"
    )
