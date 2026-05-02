"""Idea Surfacer Agent (Machine Idea Catcher).

Synthesizes research signals into actionable project ideas and writes
them to IdeaForge's ideas table (status='unscored') for downstream
scoring, classification, and Metroplex triage.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from ..claude_client import get_client
from ..config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL, IDEA_SURFACER_LOOKBACK_DAYS
from ..signal_writer import get_store  # Must import before contracts (injects sys.path)
from .ideaforge_writer import write_idea_to_ideaforge

from contracts.research_signal import ResearchSignal  # noqa: E402

logger = logging.getLogger(__name__)



def _get_recent_signals(days: int | None = None) -> list[ResearchSignal]:
    """Load research signals from the past N days with relevance >= medium.

    days=None uses IDEA_SURFACER_LOOKBACK_DAYS from config (default 14).
    """
    if days is None:
        days = IDEA_SURFACER_LOOKBACK_DAYS
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


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in text, stripping markdown fences.

    Handles Nemotron-3's common failure modes: prose preamble then JSON,
    markdown ```json fences, trailing prose after valid JSON. Uses brace-depth
    counting rather than greedy regex so nested objects parse correctly and
    trailing prose is discarded.
    """
    # Strip outermost code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n?```', text)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return None

    # Walk forward tracking brace depth, honoring string literals and escapes
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_parse_ideas_json(text: str) -> list[dict] | None:
    """Try to extract and parse ideas JSON from LLM output.

    Returns list of idea dicts on success, None on failure.
    """
    candidate = _extract_first_json_object(text)
    if candidate is None:
        logger.warning(f"No JSON object found in LLM response: {text[:200]}")
        return None

    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from LLM response: {candidate[:200]}")
        return None

    # Handle both {"ideas": [...]} envelope and bare idea object
    if "ideas" in result:
        return result["ideas"]
    if "title" in result and "description" in result:
        logger.info("Response was a bare idea object — wrapping in list")
        return [result]
    logger.warning(f"Parsed JSON but unexpected structure: {list(result.keys())}")
    return None


def _synthesize_ideas(signals: list[ResearchSignal], dry_run: bool = False) -> list[dict]:
    """Use Claude to synthesize signals into actionable project ideas.

    Returns list of dicts with: title, description, tags, signal_ids
    """
    if not signals or dry_run:
        return []

    # Cap signals to avoid blowing the context window.
    # Prioritize high-relevance signals, then sort by recency.
    MAX_SIGNALS = 75
    if len(signals) > MAX_SIGNALS:
        high = [s for s in signals if s.relevance.value == "high"]
        medium = [s for s in signals if s.relevance.value == "medium"]
        # Take all high, fill remaining with medium (most recent first)
        high.sort(key=lambda s: s.emitted_at, reverse=True)
        medium.sort(key=lambda s: s.emitted_at, reverse=True)
        signals = (high + medium)[:MAX_SIGNALS]
        logger.info(f"Capped to {len(signals)} signals ({len(high)} high, rest medium)")

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

    # Skip the LLM entirely when only one source is contributing — cross-source
    # corroboration is impossible and historical hit-rate on single-source ideas
    # is poor. Let signals roll over to the next run.
    if unique_sources < 2:
        logger.warning(
            "Only %d distinct signal source(s); skipping synthesis to preserve "
            "signals for a run with more diversity.", unique_sources
        )
        return []

    prompt = f"""You are a skill-foundry idea synthesizer. You identify MCP servers, agent skills, \
workflow tools, and pipeline components that should exist but don't yet.

The foundry builds:
- MCP servers (Python + mcp-sdk): proven pattern, 100% build success rate
- CLI tools and pip packages for developer workflows
- Agent skills and workflow pipeline components
- The build pipeline (Metroplex) can autonomously produce Python CLI tools and MCP servers

What the foundry is looking for:
- MCP servers that wrap APIs no one has wrapped yet
- Workflow tools that developers keep building from scratch
- Agent skills that would be reusable across many agent frameworks
- Pipeline components that connect existing tools in new ways

Research signals from the past week:
{chr(10).join(signal_summaries)}
{diversity_note}

IMPORTANT: Ideas that combine signals from MULTIPLE DIFFERENT sources are stronger
than ideas based on a single source. Prefer cross-source synthesis when possible.

Prioritize ideas where signals reveal a GAP (something missing) over ideas where
signals describe something that already exists.

For each idea (0-6), provide ALL of these fields:

{{
    "ideas": [
        {{
            "title": "Clear, specific project name",
            "description": "What to build and why -- 2-3 sentences covering the skill/tool vision",
            "problem_statement": "The specific gap this fills. Who needs this and what do they do today as a workaround?",
            "target_audience": "Exactly who would use this. Be specific: 'developers building Claude MCP integrations' not 'developers'",
            "tags": ["tag1", "tag2"],
            "source_signal_ids": ["signal-id-1", "signal-id-2"]
        }}
    ]
}}

Rules:
- Only suggest ideas a solo developer can MVP in 2-4 weeks
- MUST be one of: MCP server, CLI tool, pip package, agent skill, workflow component
- Each idea MUST have a non-empty problem_statement and target_audience
- If you produce 2 or more ideas, at least half MUST cite `source_signal_ids`
  drawn from 2+ distinct signal sources (the `[xxx]` tags on each signal line
  above are the sources). Single-source ideas must be the exception, not the default.
- If no clear ideas emerge from the signals, return {{"ideas": []}}
- Avoid: frontends, mobile apps, enterprise platforms, ideas requiring large teams
- Maximum 6 ideas per synthesis run"""

    client = get_client()
    response = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a JSON-only responder. Output ONLY valid JSON with no prose, no markdown fences, no explanation. Start your response with '{'."},
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    parsed = _try_parse_ideas_json(raw_text)

    if parsed is not None:
        return parsed

    # --- Retry 1: reinforce JSON-only contract. ---
    logger.warning("First LLM response was not valid JSON — retry 1 with stronger prompt")
    retry1_response = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You MUST respond with ONLY valid JSON. "
                    "No prose, no markdown, no explanation. "
                    "Just the JSON object starting with '{'. "
                    "Do not wrap in code fences."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw_text},
            {
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as JSON. "
                    "Please respond with ONLY the JSON object in the exact format "
                    'requested: {"ideas": [...]}. No other text.'
                ),
            },
        ],
    )
    retry1_text = retry1_response.choices[0].message.content.strip()
    parsed = _try_parse_ideas_json(retry1_text)
    if parsed is not None:
        logger.info("Retry 1 succeeded — parsed JSON on second attempt")
        return parsed

    # --- Retry 2: echo the failing output back, ask the model to extract just
    # the JSON from its own prior response. Nemotron-3 recovers from this when
    # the first two passes were prose-heavy. Cheap: small context delta. ---
    logger.warning("Retry 1 still invalid — retry 2 echoing failing output back")
    retry2_response = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON extraction tool. Output only the JSON object, "
                    "nothing else. Start with '{' and end with '}'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The following text was supposed to be a JSON object in the "
                    'shape {"ideas": [...]}. It failed to parse. Re-emit ONLY '
                    "the JSON object, stripping any prose, markdown fences, "
                    "explanations, or trailing text. If no coherent idea data "
                    'exists in the text, return exactly {"ideas": []}.\n\n'
                    f"--- TEXT TO EXTRACT FROM ({len(retry1_text)} chars) ---\n"
                    f"{retry1_text}\n"
                    "--- END TEXT ---"
                ),
            },
        ],
    )
    retry2_text = retry2_response.choices[0].message.content.strip()
    parsed = _try_parse_ideas_json(retry2_text)
    if parsed is not None:
        logger.info("Retry 2 succeeded — extracted JSON from failing output")
        return parsed

    logger.warning(
        "All 3 synthesis attempts failed to produce valid JSON. "
        "Last retry output (first 200 chars): %s",
        retry2_text[:200],
    )
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

    1. Load recent research signals (lookback window from config, default 14 days,
       relevance >= medium)
    2. Synthesize into 0-6 actionable project ideas via Claude
    3. Write ideas to IdeaForge (status='unscored')
    4. Mark consumed signals in ContractStore — ONLY when at least one idea was
       produced. A 0-idea run leaves signals unconsumed so they can feed the
       next attempt (guards against single-run Nemotron-3 JSON failures
       silently discarding a week of research).

    Returns summary string.
    """
    signals = _get_recent_signals()
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

    if not ideas:
        # Leave signals unconsumed so the next scheduled run can retry.
        logger.warning(
            "Synthesis produced 0 ideas from %d signals — NOT marking consumed. "
            "Signals will be available to the next run.",
            len(signals),
        )
        return f"Synthesized 0 ideas from {len(signals)} signals (signals preserved for retry)"

    written = 0

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

    # Mark ALL input signals as consumed (not just LLM-referenced ones —
    # the LLM returns made-up IDs that don't match actual signal_ids).
    all_input_ids = [s.signal_id for s in signals]
    _mark_signals_consumed(all_input_ids)
    logger.info(f"Marked {len(all_input_ids)} signals as consumed")

    return f"Synthesized {written} ideas from {len(signals)} signals"
