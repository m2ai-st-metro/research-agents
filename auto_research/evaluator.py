"""Evaluator — compares experiment results and determines winners."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import IMPROVEMENT_THRESHOLD, MIN_SIGNALS_PER_AGENT, MIN_SIGNALS_PER_EXPERIMENT
from .mini_pipeline import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class Comparison:
    """Result of comparing a variant against baseline."""
    agent: str
    param_name: str
    baseline_query: str
    variant_query: str
    baseline_ndr: float
    variant_ndr: float
    baseline_avg_score: float
    variant_avg_score: float
    baseline_signals: int
    variant_signals: int
    improvement_pct: float
    is_winner: bool
    is_valid: bool  # Had enough data for comparison
    guardrail_passed: bool  # Avg score didn't drop
    reason: str


def compare(
    agent: str,
    param_name: str,
    baseline: ExperimentResult,
    variant: ExperimentResult,
    threshold: float = IMPROVEMENT_THRESHOLD,
    min_signals: int | None = None,
) -> Comparison:
    """Compare a variant result against the baseline.

    Rules:
    1. Both must have at least min_signals relevant signals
    2. Variant non-dismiss rate must exceed baseline by threshold
    3. Avg weighted score must not drop (guardrail)
    """
    # Use per-agent override if available, else default
    if min_signals is None:
        min_signals = MIN_SIGNALS_PER_AGENT.get(agent, MIN_SIGNALS_PER_EXPERIMENT)

    # Check minimum data threshold
    if baseline.signals_relevant < min_signals:
        return Comparison(
            agent=agent,
            param_name=param_name,
            baseline_query=baseline.query,
            variant_query=variant.query,
            baseline_ndr=baseline.non_dismiss_rate,
            variant_ndr=variant.non_dismiss_rate,
            baseline_avg_score=baseline.avg_weighted_score,
            variant_avg_score=variant.avg_weighted_score,
            baseline_signals=baseline.signals_relevant,
            variant_signals=variant.signals_relevant,
            improvement_pct=0.0,
            is_winner=False,
            is_valid=False,
            guardrail_passed=True,
            reason=f"Baseline has only {baseline.signals_relevant} signals (min: {min_signals})",
        )

    if variant.signals_relevant < min_signals:
        return Comparison(
            agent=agent,
            param_name=param_name,
            baseline_query=baseline.query,
            variant_query=variant.query,
            baseline_ndr=baseline.non_dismiss_rate,
            variant_ndr=variant.non_dismiss_rate,
            baseline_avg_score=baseline.avg_weighted_score,
            variant_avg_score=variant.avg_weighted_score,
            baseline_signals=baseline.signals_relevant,
            variant_signals=variant.signals_relevant,
            improvement_pct=0.0,
            is_winner=False,
            is_valid=False,
            guardrail_passed=True,
            reason=f"Variant has only {variant.signals_relevant} signals (min: {min_signals})",
        )

    # Compute improvement
    if baseline.non_dismiss_rate > 0:
        improvement = (variant.non_dismiss_rate - baseline.non_dismiss_rate) / baseline.non_dismiss_rate
    elif variant.non_dismiss_rate > 0:
        improvement = 1.0  # Any improvement from zero is 100%
    else:
        improvement = 0.0  # Both zero

    # Check guardrail: avg score must not drop at all.
    # Even small dips indicate the variant surfaces lower-quality signals,
    # which NDR improvement alone doesn't justify.
    score_dropped = False
    if baseline.avg_weighted_score > 0 and variant.avg_weighted_score > 0:
        if variant.avg_weighted_score < baseline.avg_weighted_score:
            score_dropped = True

    is_winner = improvement >= threshold and not score_dropped

    if score_dropped:
        reason = (
            f"NDR improved {improvement:.1%} but avg score dropped "
            f"({baseline.avg_weighted_score:.1f} -> {variant.avg_weighted_score:.1f}). "
            f"Guardrail triggered."
        )
    elif improvement >= threshold:
        reason = f"Winner: NDR improved {improvement:.1%} (threshold: {threshold:.0%})"
    elif improvement > 0:
        reason = f"Improved {improvement:.1%} but below threshold ({threshold:.0%})"
    elif improvement == 0:
        reason = "No change in non-dismiss rate"
    else:
        reason = f"Regression: NDR dropped {improvement:.1%}"

    return Comparison(
        agent=agent,
        param_name=param_name,
        baseline_query=baseline.query,
        variant_query=variant.query,
        baseline_ndr=baseline.non_dismiss_rate,
        variant_ndr=variant.non_dismiss_rate,
        baseline_avg_score=baseline.avg_weighted_score,
        variant_avg_score=variant.avg_weighted_score,
        baseline_signals=baseline.signals_relevant,
        variant_signals=variant.signals_relevant,
        improvement_pct=improvement,
        is_winner=is_winner,
        is_valid=True,
        guardrail_passed=not score_dropped,
        reason=reason,
    )
