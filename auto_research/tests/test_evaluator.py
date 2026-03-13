"""Tests for the experiment evaluator."""

from auto_research.evaluator import compare
from auto_research.mini_pipeline import ExperimentResult


def _make_result(
    query: str = "test",
    signals_relevant: int = 10,
    ideas_total: int = 3,
    ideas_non_dismissed: int = 1,
    avg_weighted_score: float = 5.0,
) -> ExperimentResult:
    """Create a test ExperimentResult."""
    ndr = ideas_non_dismissed / ideas_total if ideas_total > 0 else 0.0
    return ExperimentResult(
        query=query,
        signals_found=signals_relevant + 5,
        signals_relevant=signals_relevant,
        ideas_synthesized=ideas_total,
        ideas_non_dismissed=ideas_non_dismissed,
        ideas_total=ideas_total,
        non_dismiss_rate=ndr,
        avg_weighted_score=avg_weighted_score,
    )


def test_clear_winner():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=1)  # 10% NDR
    variant = _make_result(ideas_total=10, ideas_non_dismissed=3)   # 30% NDR

    result = compare("arxiv", "test_param", baseline, variant)
    assert result.is_winner
    assert result.is_valid
    assert result.guardrail_passed
    assert abs(result.improvement_pct - 2.0) < 0.01  # 200% improvement


def test_below_threshold():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=1)  # 10% NDR
    variant = _make_result(ideas_total=10, ideas_non_dismissed=1)   # 10% NDR (same)

    result = compare("arxiv", "test_param", baseline, variant)
    assert not result.is_winner
    assert result.is_valid
    assert result.improvement_pct == 0.0


def test_regression():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=3)  # 30% NDR
    variant = _make_result(ideas_total=10, ideas_non_dismissed=1)   # 10% NDR

    result = compare("arxiv", "test_param", baseline, variant)
    assert not result.is_winner
    assert result.is_valid
    assert result.improvement_pct < 0


def test_insufficient_baseline_signals():
    baseline = _make_result(signals_relevant=2)  # Too few
    variant = _make_result(signals_relevant=10)

    result = compare("arxiv", "test_param", baseline, variant, min_signals=5)
    assert not result.is_winner
    assert not result.is_valid
    assert "Baseline has only 2" in result.reason


def test_insufficient_variant_signals():
    baseline = _make_result(signals_relevant=10)
    variant = _make_result(signals_relevant=3)  # Too few

    result = compare("arxiv", "test_param", baseline, variant, min_signals=5)
    assert not result.is_winner
    assert not result.is_valid


def test_guardrail_blocks_score_drop():
    """Winner by NDR, but avg score dropped — should be blocked."""
    baseline = _make_result(
        ideas_total=10, ideas_non_dismissed=1,
        avg_weighted_score=6.0,
    )
    variant = _make_result(
        ideas_total=10, ideas_non_dismissed=3,
        avg_weighted_score=4.0,  # Big score drop
    )

    result = compare("arxiv", "test_param", baseline, variant)
    assert not result.is_winner  # Blocked by guardrail
    assert result.is_valid
    assert not result.guardrail_passed
    assert "Guardrail" in result.reason


def test_both_zero_ndr():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=0)
    variant = _make_result(ideas_total=10, ideas_non_dismissed=0)

    result = compare("arxiv", "test_param", baseline, variant)
    assert not result.is_winner
    assert result.is_valid
    assert result.improvement_pct == 0.0


def test_improvement_from_zero():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=0)  # 0% NDR
    variant = _make_result(ideas_total=10, ideas_non_dismissed=2)   # 20% NDR

    result = compare("arxiv", "test_param", baseline, variant)
    assert result.is_winner
    assert result.improvement_pct == 1.0  # 100%


def test_custom_threshold():
    baseline = _make_result(ideas_total=10, ideas_non_dismissed=1)
    variant = _make_result(ideas_total=10, ideas_non_dismissed=2)  # 100% improvement

    # High threshold
    result = compare("arxiv", "test_param", baseline, variant, threshold=1.5)
    assert not result.is_winner  # 100% < 150%

    # Low threshold
    result = compare("arxiv", "test_param", baseline, variant, threshold=0.5)
    assert result.is_winner  # 100% > 50%
