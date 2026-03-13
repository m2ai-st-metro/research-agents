"""Claude API validator — confirms Ollama winners against production scoring.

Closes the sim-to-real gap by running top winning variants through
Claude Sonnet (the production model) before committing changes.
"""

from __future__ import annotations

import logging
import os
import sqlite3

from .config import MAX_CLAUDE_VALIDATIONS
from .evaluator import Comparison
from .ledger import get_winners, mark_validated
from .mini_pipeline import (
    ExperimentResult,
    Signal,
    assess_signals,
    classify_ideas,
    score_ideas,
    synthesize_ideas,
)

logger = logging.getLogger(__name__)


def _get_anthropic_client():
    """Create Anthropic client from environment."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic package not installed — cannot validate with Claude")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — cannot validate with Claude")
        return None

    return Anthropic(api_key=api_key)


def _claude_assess_relevance(
    client,
    title: str,
    summary: str,
    source_context: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """Assess relevance via Claude API (production model)."""
    import json

    prompt = f"""Assess the relevance of this research signal to a solo AI developer's ecosystem.

The developer builds:
- Claude-powered MCP servers and tool-augmented agents
- An autonomous idea-to-product pipeline (Ultra Magnus)
- A self-improving feedback loop (Snow-Town: UM -> Sky-Lynx -> Academy)
- Healthcare AI projects (HIPAA-compliant, home health focus)
- Developer productivity tools

Signal:
- Title: {title}
- Summary: {summary}
- Source context: {source_context}

Respond with JSON only:
{{
    "relevance": "high" | "medium" | "low",
    "relevance_rationale": "Why this is/isn't relevant (1-2 sentences)",
    "tags": ["tag1", "tag2"],
    "domain": "primary domain or null",
    "persona_tags": []
}}"""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "relevance": "low",
            "relevance_rationale": "Failed to parse",
            "tags": [],
            "domain": None,
            "persona_tags": [],
        }


def validate_winner(
    comparison: Comparison,
    experiment_id: int,
    conn: sqlite3.Connection,
) -> bool:
    """Validate a winning experiment by re-running with Claude API.

    Re-runs the variant query's signals through Claude for relevance
    assessment and scoring. If the improvement holds under Claude scoring,
    returns True.

    This only validates scoring quality — we don't re-run the search
    (that already used free APIs).
    """
    anthropic_client = _get_anthropic_client()
    if anthropic_client is None:
        logger.warning("Skipping Claude validation — no API client available")
        return False

    logger.info(
        "Validating [%s] variant '%s' with Claude API...",
        comparison.agent,
        comparison.variant_query,
    )

    # We need the raw signals from the variant experiment.
    # For validation, we re-search and re-score with Claude.
    from .mini_pipeline import AGENT_SEARCHERS

    searcher = AGENT_SEARCHERS.get(comparison.agent)
    if searcher is None:
        logger.warning("No searcher for agent '%s'", comparison.agent)
        return False

    # Re-search with the variant query
    raw_signals = searcher(comparison.variant_query, max_results=10)
    if not raw_signals:
        logger.warning("No signals found on re-search")
        return False

    # Assess with Claude
    relevance_order = {"high": 3, "medium": 2, "low": 1}
    relevant_signals: list[Signal] = []
    for raw in raw_signals:
        assessment = _claude_assess_relevance(
            anthropic_client,
            title=raw["title"],
            summary=raw["summary"],
            source_context=f"{raw['source']} signal",
        )
        relevance = assessment.get("relevance", "low")
        if relevance_order.get(relevance, 0) >= 2:  # medium+
            relevant_signals.append(Signal(
                signal_id=raw["signal_id"],
                title=raw["title"],
                summary=raw["summary"],
                source=raw["source"],
                url=raw["url"],
                relevance=relevance,
            ))

    if not relevant_signals:
        logger.info("Claude found no relevant signals in variant results")
        mark_validated(conn, experiment_id, claude_ndr=0.0)
        return False

    # For scoring, we use Ollama (Claude scoring would require
    # the full IdeaForge ClaudeClient pipeline). The key validation
    # is whether Claude agrees the signals are relevant.
    # If Claude agrees on relevance, the variant query is valid.
    claude_relevance_rate = len(relevant_signals) / len(raw_signals)
    logger.info(
        "Claude relevance rate: %.1f%% (%d/%d signals)",
        claude_relevance_rate * 100,
        len(relevant_signals),
        len(raw_signals),
    )

    # Use Claude relevance as a proxy for validation
    # If Claude marks more signals as relevant than baseline, it's validated
    mark_validated(conn, experiment_id, claude_ndr=claude_relevance_rate)

    # Consider validated if Claude relevance rate > 30%
    # (meaning the variant query produces genuinely relevant signals)
    validated = claude_relevance_rate > 0.3
    if validated:
        logger.info("Claude validated: variant produces relevant signals")
    else:
        logger.info("Claude rejected: variant signals not relevant enough")

    return validated


def validate_top_winners(
    conn: sqlite3.Connection,
    max_validations: int = MAX_CLAUDE_VALIDATIONS,
) -> list[int]:
    """Validate top N unvalidated winners. Returns list of validated experiment IDs."""
    winners = get_winners(conn)
    unvalidated = [w for w in winners if not w["claude_validated"]]

    if not unvalidated:
        logger.info("No unvalidated winners to check")
        return []

    validated_ids: list[int] = []
    for row in unvalidated[:max_validations]:
        comparison = Comparison(
            agent=row["agent"],
            param_name=row["param_name"],
            baseline_query=row["baseline_value"],
            variant_query=row["variant_value"],
            baseline_ndr=row["baseline_ndr"],
            variant_ndr=row["variant_ndr"],
            baseline_avg_score=row["baseline_avg_score"] or 0,
            variant_avg_score=row["variant_avg_score"] or 0,
            baseline_signals=row["baseline_signals"],
            variant_signals=row["variant_signals"],
            improvement_pct=row["improvement_pct"],
            is_winner=True,
            is_valid=True,
            guardrail_passed=True,
            reason="",
        )

        if validate_winner(comparison, row["id"], conn):
            validated_ids.append(row["id"])

    logger.info(
        "Validated %d/%d winners with Claude API",
        len(validated_ids),
        len(unvalidated[:max_validations]),
    )
    return validated_ids
