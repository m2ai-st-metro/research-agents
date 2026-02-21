"""Idea Surfacer Agent (Machine Idea Catcher).

Synthesizes research signals into actionable project ideas and writes
them directly to Ultra-Magnus's caught_ideas.db.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from ..claude_client import get_client
from ..config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL, ULTRA_MAGNUS_DB
from ..signal_writer import get_store

from contracts.research_signal import ResearchSignal  # noqa: E402

logger = logging.getLogger(__name__)

# Schema for caught_ideas table (bootstrapped if needed)
CAUGHT_IDEAS_SCHEMA = """
CREATE TABLE IF NOT EXISTS caught_ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    source_context TEXT,
    caught_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'processed', 'failed')),
    processed_at TIMESTAMP,
    factory_id TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);
"""


def _get_um_db_path() -> Path:
    """Get Ultra-Magnus caught_ideas.db path."""
    env_path = os.environ.get("IDEA_CATCHER_DB")
    if env_path:
        return Path(env_path)
    return ULTRA_MAGNUS_DB


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
    for s in signals:
        signal_summaries.append(
            f"- [{s.source.value}] {s.title}: {s.summary} "
            f"(relevance: {s.relevance.value}, domain: {s.domain or 'general'})"
        )

    prompt = f"""You are a project idea synthesizer for a solo AI developer. Given these research signals, identify 0-3 actionable project ideas.

The developer's ecosystem:
- Claude-powered MCP servers and tool-augmented agents
- An autonomous idea-to-product pipeline (Ultra Magnus)
- A self-improving feedback loop (Snow-Town)
- Healthcare AI projects (HIPAA-compliant, home health focus)
- Developer productivity tools

Research signals from the past week:
{chr(10).join(signal_summaries)}

For each idea, provide:
1. A clear, actionable title
2. A 2-3 sentence description of what to build and why
3. Tags for categorization
4. Which signal IDs inspired this idea

Respond with JSON only:
{{
    "ideas": [
        {{
            "title": "Project title",
            "description": "What to build and why (2-3 sentences)",
            "tags": ["tag1", "tag2"],
            "source_signal_ids": ["signal-id-1", "signal-id-2"]
        }}
    ]
}}

Rules:
- Only suggest ideas that are actionable for a solo developer
- Prefer ideas that leverage existing infrastructure (MCP, Claude API, Snow-Town)
- If no clear ideas emerge from the signals, return {{"ideas": []}}
- Maximum 3 ideas per synthesis run"""

    client = get_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
        return result.get("ideas", [])
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse synthesis response: {text[:200]}")
        return []


def _write_idea_to_um(
    title: str,
    description: str,
    tags: list[str],
    db_path: Path | None = None,
) -> int:
    """Write an idea directly to Ultra-Magnus's caught_ideas.db.

    Returns the inserted idea ID.
    """
    path = db_path or _get_um_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        # Ensure table exists
        conn.executescript(CAUGHT_IDEAS_SCHEMA)

        # Add machine-surfaced tag
        all_tags = list(set(tags + ["machine-surfaced"]))

        cursor = conn.execute(
            "INSERT INTO caught_ideas (title, raw_content, tags, source_context) VALUES (?, ?, ?, ?)",
            (
                title,
                description,
                json.dumps(all_tags),
                "research-agents:idea-surfacer",
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


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
    3. Write ideas to caught_ideas.db
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

    for idea in ideas:
        um_id = _write_idea_to_um(
            title=idea["title"],
            description=idea["description"],
            tags=idea.get("tags", []),
        )
        logger.info(f"Wrote idea #{um_id}: {idea['title']}")
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
