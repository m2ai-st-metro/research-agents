"""Idea Surfacer Agent (Machine Idea Catcher).

Synthesizes research signals into actionable project ideas and writes
them to IdeaForge's ideas table (status='unscored') for downstream
scoring, classification, and Metroplex triage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from contracts.research_signal import ResearchSignal  # noqa: E402

from ..claude_client import get_client
from ..config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from ..signal_writer import get_store
from .ideaforge_writer import write_idea_to_ideaforge

logger = logging.getLogger(__name__)



def _get_recent_signals(days: int = 7) -> list[ResearchSignal]:
    """Load research signals from the past N days with relevance >= medium."""
    store = get_store()
    try:
        # Query unconsumed signals
        all_signals = store.query_signals(consumed=False, limit=500)

        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            s for s in all_signals
            if s.emitted_at >= cutoff and s.relevance.value in ("high", "medium")
        ]
        return recent
    finally:
        store.close()


def _synthesize_ideas(signals: list[ResearchSignal], dry_run: bool = False) -> list[dict]:
    """Use Claude to synthesize signals into actionable project ideas.

    Returns list of dicts with: title, description, tags, signal_ids
    """
    if not signals or dry_run:
        return []

    # Format signals for the prompt
    signal_summaries = []
    source_counts: dict[str, int] = {}
    for s in signals:
        signal_summaries.append(
            f"- [{s.source.value}] {s.title}: {s.summary} "
            f"(relevance: {s.relevance.value}, domain: {s.domain or 'general'})"
        )
        source_counts[s.source.value] = source_counts.get(s.source.value, 0) + 1

    # Compute source diversity summary
    unique_sources = len(source_counts)
    diversity_note = (
        f"\nSignal diversity: {unique_sources} distinct sources "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(source_counts.items()))})"
    )

    prompt = f"""You are a project idea synthesizer for an AI engineer who builds and sells developer tools autonomously.

The developer's current ecosystem and strengths:
- MCP servers (Python + mcp-sdk): proven build pattern, 100% success rate
- Claude Agent SDK: multi-agent autonomous builder (YCE Harness)
- Metroplex: L5 autonomy coordinator that triages, builds, and publishes projects
- Command Center (CMD): web UI for dispatching agents (Soundwave, Ravage, Content)
- Galvatron: Telegram bot (Claude Code backend) for ops and quick tasks
- Sky-Lynx: self-improvement agent that tunes the ecosystem semi-weekly
- M2AI VoiceBots: commercial VA service targeting agencies/SMBs
- Starscream: LinkedIn AI thought leadership content engine

Target market: solo developers and small teams using AI agents, MCP, and LLM tooling.
Revenue model: open-source tools with paid tiers, productized services, consulting.

Research signals from the past week:
{chr(10).join(signal_summaries)}
{diversity_note}

IMPORTANT: Ideas that combine signals from MULTIPLE DIFFERENT sources are stronger
than ideas based on a single source. Prefer cross-source synthesis when possible.

For each idea (0-3), provide ALL of these fields:

{{
    "ideas": [
        {{
            "title": "Clear, specific project name",
            "description": "What to build and why — 2-3 sentences covering the product vision",
            "problem_statement": "The specific pain point this solves. Who has this problem and how badly? What do they do today as a workaround?",
            "target_audience": "Exactly who would pay for or use this. Be specific: 'solo AI developers shipping MCP servers' not 'developers'",
            "tags": ["tag1", "tag2"],
            "source_signal_ids": ["signal-id-1", "signal-id-2"]
        }}
    ]
}}

Rules:
- Only suggest ideas a solo developer can MVP in 2-4 weeks
- Strong preference for: MCP servers, CLI tools, developer productivity, agent tooling
- Each idea MUST have a non-empty problem_statement and target_audience
- If no clear ideas emerge from the signals, return {{"ideas": []}}
- Avoid: healthcare compliance, enterprise platforms, ideas requiring large teams
- Maximum 3 ideas per synthesis run"""

    client = get_client()
    response = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()

    # Robust JSON extraction: handle markdown fences and/or prose preamble
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    else:
        # Claude may prefix the JSON with prose — extract the first JSON object/array
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)

    try:
        result = json.loads(text)
        return result.get("ideas", [])
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse synthesis response: {text[:200]}")
        return []


def _mark_signals_consumed(signal_ids: list[str]) -> None:
    """Mark signals as consumed by the idea surfacer."""
    store = get_store()
    try:
        for signal_id in signal_ids:
            store.update_signal_consumed_by(signal_id, "idea-surfacer")
    finally:
        store.close()


def run_agent(dry_run: bool = False) -> str:
    """Run the idea surfacer agent.

    1. Load recent research signals (past week, relevance >= medium)
    2. Synthesize into 0-3 actionable project ideas via Claude
    3. Write ideas to IdeaForge (status='unscored')
    4. Mark consumed signals in ContractStore

    Returns summary string.
    """
    signals = _get_recent_signals(days=7)
    logger.info(f"Found {len(signals)} recent unconsumed signals")

    if not signals:
        return "No recent unconsumed signals to synthesize"

    if dry_run:
        logger.info(f"[DRY RUN] Would synthesize from {len(signals)} signals:")
        for s in signals[:10]:
            logger.info(f"  [{s.source.value}] {s.title}")
        return f"[DRY RUN] {len(signals)} signals available for synthesis"

    ideas = _synthesize_ideas(signals)
    logger.info(f"Synthesized {len(ideas)} ideas")

    written = 0
    consumed_signal_ids: list[str] = []

    # Build signal ID -> source mapping for provenance
    signal_source_map: dict[str, str] = {}
    for s in signals:
        signal_source_map[s.signal_id] = s.source.value

    for idea in ideas:
        # Determine primary signal source from the idea's source signals
        idea_signal_ids = idea.get("source_signal_ids", [])
        primary_source = "idea_surfacer"
        if idea_signal_ids:
            first_source = signal_source_map.get(idea_signal_ids[0])
            if first_source:
                primary_source = first_source

        idea_id = write_idea_to_ideaforge(
            title=idea["title"],
            description=idea["description"],
            tags=idea.get("tags", []),
            source_signal_ids=idea_signal_ids,
            problem_statement=idea.get("problem_statement", ""),
            target_audience=idea.get("target_audience", ""),
            signal_source=primary_source,
        )
        logger.info(f"Wrote idea #{idea_id} to IdeaForge: {idea['title']}")
        written += 1

        # Track which signals were consumed
        for sid in idea.get("source_signal_ids", []):
            if sid not in consumed_signal_ids:
                consumed_signal_ids.append(sid)

    # Mark consumed signals
    if consumed_signal_ids:
        _mark_signals_consumed(consumed_signal_ids)
        logger.info(f"Marked {len(consumed_signal_ids)} signals as consumed")

    return f"Synthesized {written} ideas from {len(signals)} signals"
