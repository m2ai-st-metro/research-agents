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
from ..config import (
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    IDEA_SURFACER_FALLBACK_MODEL,
    IDEA_SURFACER_LOOKBACK_DAYS,
)
from ..signal_writer import get_store  # Must import before contracts (injects sys.path)
from .ideaforge_writer import write_idea_to_ideaforge

from contracts.research_signal import ResearchSignal  # noqa: E402

logger = logging.getLogger(__name__)


class TruncatedJSONError(ValueError):
    """Raised when text contains an opening JSON brace that never closes.

    Distinguishes "no JSON at all" (parser returns None) from "JSON started
    but the LLM response was cut off mid-string". Callers can branch on this
    to log accurately and decide whether to retry rather than discarding the
    response as unparseable prose.
    """



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
    """Return the first parseable balanced JSON object in text.

    Walks ALL `{` positions (after stripping outermost markdown fences). For
    each candidate, runs a brace-depth scan honoring string literals and
    escapes; if depth returns to 0 and `json.loads` accepts the slice, that
    slice wins. Adversarial chain-of-thought prose like "schema looks like
    {key: value}" no longer hijacks extraction — the next real JSON object
    in the text gets a chance.

    Behaviors:
      - Returns first candidate substring that round-trips through json.loads.
      - Raises TruncatedJSONError if at least one candidate's depth-walk runs
        off the end of the string without closing AND no later candidate
        parses (LLM response was cut off mid-JSON).
      - Returns None if no `{` is present at all.
    """
    # Strip outermost code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n?```', text)
    if fence_match:
        text = fence_match.group(1)

    if "{" not in text:
        return None

    saw_truncated_candidate = False
    n = len(text)

    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        closed_at: int | None = None

        for i in range(start, n):
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
                    closed_at = i
                    break

        if closed_at is None:
            # Walked to end of text without depth returning to 0 — truncated.
            saw_truncated_candidate = True
            continue

        candidate = text[start:closed_at + 1]
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            # Stray-brace prose like "{key: value, nested: {inner: thing}}"
            # closes balanced but isn't real JSON. Try the next `{`.
            continue
        return candidate

    if saw_truncated_candidate:
        raise TruncatedJSONError(
            "JSON started but truncated: opening brace found, but no balanced "
            "close before end of text and no later candidate parsed."
        )
    return None


def _try_parse_ideas_json(text: str) -> list[dict] | None:
    """Try to extract and parse ideas JSON from LLM output.

    Returns list of idea dicts on success, None on failure. Logs distinguish
    truncation (JSON started but cut off) from no-JSON-at-all so operators
    can tell a context-window blowout from a prose-only response.
    """
    try:
        candidate = _extract_first_json_object(text)
    except TruncatedJSONError as exc:
        logger.warning(
            "LLM response appears truncated (JSON started but never closed): "
            "%s | head: %s",
            exc,
            text[:200],
        )
        return None

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

    # Soft diversity check: with the 2026-05-08 life-domain pivot, Reddit is the
    # only active life-domain ingestion source. The cross-source corroboration
    # rule no longer applies as a hard gate -- within-Reddit subreddit diversity
    # is the equivalent signal. The prompt-level "prefer 2+ distinct sources"
    # rule still nudges multi-source synthesis whenever diversity exists.
    if unique_sources < 2:
        logger.info(
            "Single-source synthesis (%d sources, %d signals) — life-domain "
            "Reddit-only era. Proceeding without cross-source corroboration.",
            unique_sources, len(signals)
        )

    prompt = f"""You are a life-domain idea synthesizer. From raw human-life research \
signals, you identify SCENES — concrete moments in a person's day where coordination, \
admin, decision-fatigue, or caregiving load is crushing them, and where a patient \
AI companion could plausibly absorb 60-70% of that load (the "house manager" benchmark: \
what wealthy households hire a human to handle, the rest of us juggle alone).

You are NOT looking for developer tools, MCP servers, CLI utilities, agent SDKs, or \
APIs to wrap. You are looking for moments in the lives of middle-income, non-technical \
people — aging-parent caregivers, parents of newborns or autistic kids, people stuck in \
insurance phone trees, people navigating chronic illness or menopause — that an AI \
companion could meaningfully ease.

Calibration example of a well-formed Scene-shaped idea:
{{
    "title": "Wellness Copilot",
    "description": "An always-available personal health guide that helps a parent triage a sick child at 2 AM, decide whether to call the pediatrician or wait until morning, and keep a running log of symptoms across the week.",
    "problem_statement": "It's 2 AM. A parent is awake with a feverish toddler, scrolling through three different symptom-checker sites, second-guessing whether 102.3 warrants the ER. They've already had this exact night four times this winter, and tomorrow they'll be expected to function at work. 27% of personal guidance conversations people have with AI today are health-related — the demand is enormous, the trusted answer is missing.",
    "target_audience": "Middle-income parents of young children, especially first-time parents and single parents, who don't have a doctor in the family and can't afford concierge medicine.",
    "struggling_user": "I'm awake at 2 AM with a sick kid, terrified of overreacting and terrified of underreacting, and I have nobody to ask except a search engine.",
    "weight_hint": "~6 nights per winter of lost sleep per parent; one unnecessary ER visit averages $1,200 out-of-pocket; an estimated 40+ hours/year per family lost to symptom-checker rabbit holes.",
    "agentic_relief": "An AI companion could maintain the child's symptom history, run a structured triage conversation calibrated to the family's pediatrician's actual escalation thresholds, and produce a one-page summary the parent can hand to a clinician. It cannot diagnose, but it can absorb the mental load of remembering, comparing, and deciding when to escalate — the 60-70% of cognitive work that isn't the medical judgment itself.",
    "tags": ["pediatric", "caregiving", "after-hours-triage"],
    "source_signal_ids": ["signal-id-1", "signal-id-2"]
}}

Anti-patterns — REJECT outputs that look like this:
- "wellness-cli", "health-mcp-server", "symptom-checker-agent-skill" — these are tool names, not Scene names. Use "Wellness Copilot" style.
- "Many people struggle with healthcare admin..." — generic openers are forbidden. Name the specific 2 AM, the specific person, the specific decision.
- "Users can leverage AI to streamline..." — corporate-speak. Write like you're describing a real night in a real apartment.
- Tech tags like "MCP", "CLI", "agent-sdk", "API", "SDK" — replaced by life-domain tags ("caregiving", "insurance-navigation", "newborn-care", "menopause", "chronic-pain", "elder-care", etc.).

Persona defaults (override only if signals demand otherwise):
- Middle-income (not ultra-wealthy, not destitute)
- Non-technical (does not write code, does not run CLIs)
- Time-starved, sleep-deprived, or both
- If a signal is about ultra-wealthy people, reframe: "what house managers do for the rich, this would do for everyone else."

Research signals from the past week:
{chr(10).join(signal_summaries)}
{diversity_note}

Prefer ideas that combine signals from MULTIPLE DIFFERENT sources — cross-source \
corroboration means the Scene is real, not a one-off vent post.

For each idea (0-6), provide ALL of these fields:

{{
    "ideas": [
        {{
            "title": "Short concrete name (e.g. 'Wellness Copilot', 'Insurance Navigator'). NOT a tool/CLI/MCP name.",
            "description": "1-2 sentences in plain English describing what this would do for a person living the scene.",
            "problem_statement": "2-3 sentences describing the LIVED STRUGGLE. Specific persona, specific moment, specific daily reality. No 'many people'.",
            "target_audience": "1-2 sentences naming WHO lives this problem. Concrete: 'Middle-income parents juggling work and elementary-school kids' health logistics' — not 'users' or 'parents'.",
            "struggling_user": "Single sentence in first person from the user's POV — like a quote you'd read in an Atlantic feature.",
            "weight_hint": "At least one concrete cost number — hours/week, dollars/year, decision count, missed-sleep nights, etc. No 'a lot of time' — give a number even if it's an estimate.",
            "agentic_relief": "2-3 sentences on what an AI companion could plausibly handle (target ~60-70% of the cognitive/coordination load — the house-manager benchmark). Bounded and specific. No 'AI streamlines everything' magic-handwaving.",
            "tags": ["life-domain-tag-1", "life-domain-tag-2"],
            "source_signal_ids": ["signal-id-1", "signal-id-2"]
        }}
    ]
}}

Rules:
- ALL eight content fields (title, description, problem_statement, target_audience, struggling_user, weight_hint, agentic_relief, tags) MUST be non-empty.
- Tags MUST be life-domain only — caregiving, pediatric, insurance-navigation, elder-care, autism-parenting, postpartum, sleep-training, chronic-illness, menopause, mental-health, etc. Reject tech tags.
- If you produce 2+ ideas, at least half MUST cite source_signal_ids drawn from 2+ distinct signal sources.
- Quality over quantity — return fewer than 3 ideas (or zero) if the signals don't support more. Empty {{"ideas": []}} is acceptable.
- Maximum 6 ideas per synthesis run."""

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
        "All 3 primary-model synthesis attempts failed to produce valid JSON. "
        "Last retry output (first 200 chars): %s",
        retry2_text[:200],
    )

    # --- Fallback hop: one final attempt against IDEA_SURFACER_FALLBACK_MODEL.
    # Empty default = no fallback (preserves pre-resilience behavior). When
    # set, fire ONE call with the fallback model id, reusing retry-2's
    # "extract JSON from failing output" framing. This mirrors the
    # metroplex-spec-expander-fallback pattern. ---
    if IDEA_SURFACER_FALLBACK_MODEL:
        logger.warning(
            "Hopping to fallback model %r after primary exhaustion",
            IDEA_SURFACER_FALLBACK_MODEL,
        )
        fallback_response = client.chat.completions.create(
            model=IDEA_SURFACER_FALLBACK_MODEL,
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
                        'shape {"ideas": [...]}. The primary model failed to produce '
                        "valid JSON across three attempts. Re-emit ONLY the JSON "
                        "object, stripping any prose, markdown fences, explanations, "
                        "or trailing text. If no coherent idea data exists in the "
                        'text, return exactly {"ideas": []}.\n\n'
                        f"--- TEXT TO EXTRACT FROM ({len(retry2_text)} chars) ---\n"
                        f"{retry2_text}\n"
                        "--- END TEXT ---"
                    ),
                },
            ],
        )
        fallback_text = fallback_response.choices[0].message.content.strip()
        parsed = _try_parse_ideas_json(fallback_text)
        if parsed is not None:
            logger.info(
                "Fallback model %r succeeded after primary exhaustion",
                IDEA_SURFACER_FALLBACK_MODEL,
            )
            return parsed
        logger.warning(
            "Fallback model %r also failed to produce valid JSON",
            IDEA_SURFACER_FALLBACK_MODEL,
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
            struggling_user=idea.get("struggling_user", ""),
            weight_hint=idea.get("weight_hint", ""),
            agentic_relief=idea.get("agentic_relief", ""),
            scoring_rubric="life_domain",
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
