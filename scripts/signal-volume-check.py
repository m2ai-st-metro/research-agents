#!/usr/bin/env python3
"""
Signal volume impact check for the 2026-05-09 tech-source kill.

Compares the 24h period after the kill against the prior 7-day average,
per source. Writes a markdown report to vault/daily/.

Usage:
  signal-volume-check.py            # Write report to vault
  signal-volume-check.py --dry-run  # Print to stdout, no write
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB = Path.home() / "projects" / "st-records" / "data" / "persona_metrics.db"
IDEAFORGE_DB = Path.home() / "projects" / "ideaforge" / "data" / "ideaforge.db"
VAULT_DIR = Path.home() / "vault" / "daily"
KILL_INSTANT = "2026-05-09T00:55:00+00:00"  # cron RELOAD time


def fmt_pct(new: float, base: float) -> str:
    if base == 0:
        return "n/a (no baseline)" if new == 0 else "+inf% (was 0)"
    pct = (new - base) / base * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ref-time", help="Override reference time (ISO8601)")
    args = parser.parse_args()

    ref_dt = (
        datetime.fromisoformat(args.ref_time)
        if args.ref_time
        else datetime.now(timezone.utc)
    )

    window_24h_start = (ref_dt - timedelta(hours=24)).isoformat()
    window_24h_end = ref_dt.isoformat()
    window_7d_start = (ref_dt - timedelta(days=8)).isoformat()
    window_7d_end = (ref_dt - timedelta(hours=24)).isoformat()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Sources active in the prior 7 days
    cur.execute(
        """
        SELECT source, COUNT(*)
        FROM research_signals
        WHERE emitted_at >= ? AND emitted_at < ?
        GROUP BY source
        """,
        (window_7d_start, window_7d_end),
    )
    prior_7d = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute(
        """
        SELECT source, COUNT(*)
        FROM research_signals
        WHERE emitted_at >= ? AND emitted_at < ?
        GROUP BY source
        """,
        (window_24h_start, window_24h_end),
    )
    last_24h = {row[0]: row[1] for row in cur.fetchall()}

    all_sources = sorted(set(prior_7d) | set(last_24h))

    lines = [
        "---",
        f"date: {ref_dt.date().isoformat()}",
        "tags: [research-agents, signal-volume, life-domain-pivot, tech-kill]",
        "type: pipeline-impact-report",
        "---",
        "",
        "# Research Signal Volume — 24h Impact of 2026-05-09 Tech-Source Kill",
        "",
        f"**Reference time**: {ref_dt.isoformat()}",
        f"**Kill instant** (cron RELOAD): {KILL_INSTANT}",
        f"**Last-24h window**: {window_24h_start} → {window_24h_end}",
        f"**Prior-7d window**: {window_7d_start} → {window_7d_end}",
        "",
        "## Per-source volume",
        "",
        "| Source | Last 24h | Prior 7d total | Prior 7d avg/day | Δ vs avg | Status |",
        "|---|---:|---:|---:|---|---|",
    ]

    expected_off = {
        "tool_monitor",
        "rss_scanner",
        "trend_analyzer",
        "youtube_scanner",
        "gemini_research",
        "perplexity",
        "chatgpt",
    }
    # idea_surfacer writes ideas to ideaforge.db, not research_signals — checked separately below
    expected_on = {"reddit"}

    for src in all_sources:
        recent = last_24h.get(src, 0)
        prior_total = prior_7d.get(src, 0)
        prior_avg = prior_total / 7
        delta = fmt_pct(recent, prior_avg)
        if src in expected_off:
            status = "OK (killed)" if recent == 0 else "LEAK — still emitting"
        elif src in expected_on:
            status = "OK (kept)" if recent > 0 else "STALL — should be running"
        else:
            status = "unexpected source"
        lines.append(
            f"| `{src}` | {recent} | {prior_total} | {prior_avg:.1f} | {delta} | {status} |"
        )

    total_recent = sum(last_24h.values())
    total_prior_avg = sum(prior_7d.values()) / 7
    lines += [
        "",
        "## Totals",
        "",
        f"- Last 24h: **{total_recent}** signals",
        f"- Prior 7d avg/day: **{total_prior_avg:.1f}** signals",
        f"- Net delta: **{fmt_pct(total_recent, total_prior_avg)}**",
        "",
    ]

    # idea_surfacer downstream check (writes to ideaforge.db, not research_signals)
    try:
        ideaforge = sqlite3.connect(IDEAFORGE_DB)
        cur2 = ideaforge.cursor()
        cur2.execute(
            "SELECT COUNT(*) FROM ideas WHERE synthesized_at >= ? AND synthesized_at < ?",
            (window_24h_start, window_24h_end),
        )
        ideas_24h = cur2.fetchone()[0]
        cur2.execute(
            "SELECT COUNT(*) FROM ideas WHERE synthesized_at >= ? AND synthesized_at < ?",
            (window_7d_start, window_7d_end),
        )
        ideas_7d = cur2.fetchone()[0]
        ideaforge.close()
        ideas_avg = ideas_7d / 7
        lines += [
            "## idea_surfacer (downstream — ideaforge.db)",
            "",
            f"- Ideas synthesized last 24h: **{ideas_24h}**",
            f"- Prior 7d avg/day: **{ideas_avg:.1f}**",
            f"- Net delta: **{fmt_pct(ideas_24h, ideas_avg)}**",
            "",
        ]
        if ideas_24h == 0 and ideas_avg > 0:
            lines.append(
                "**STALL**: `idea_surfacer` synthesized 0 ideas in last 24h — "
                "Reddit signal may be too thin for daily synthesis. Check pipeline.log.\n"
            )
    except Exception as e:
        lines.append(f"_idea_surfacer check skipped: {e}_\n")

    lines += ["## Read", ""]

    leaks = [s for s in expected_off if last_24h.get(s, 0) > 0]
    stalls = [s for s in expected_on if last_24h.get(s, 0) == 0]

    if not leaks and not stalls:
        lines.append("All 7 killed sources at 0; both kept sources active. Cron change behaving as designed.")
    if leaks:
        lines.append(
            f"**LEAK**: {', '.join('`'+s+'`' for s in leaks)} still emitting after 2026-05-09 kill. "
            "Investigate: deploy template vs installed cron, or another scheduler firing them."
        )
    if stalls:
        lines.append(
            f"**STALL**: {', '.join('`'+s+'`' for s in stalls)} produced 0 signals in last 24h — should be running. "
            "Check `/var/log/research-agents/pipeline.log` for errors."
        )

    if total_recent < total_prior_avg * 0.1:
        lines.append(
            f"\n**Volume warning**: Last-24h volume is <10% of prior 7d avg. "
            f"life_domain ingestion may be too thin for `idea_surfacer` to synthesize daily. "
            f"If sustained, fan-out to a second domain (D2) becomes urgent."
        )

    output = "\n".join(lines) + "\n"

    if args.dry_run:
        print(output)
        return 0

    out_path = VAULT_DIR / f"{ref_dt.date().isoformat()}-research-signal-volume-impact.md"
    out_path.write_text(output)
    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
