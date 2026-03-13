"""Tests for the mini-pipeline classification and scoring logic."""

from auto_research.mini_pipeline import Idea, classify_ideas, ExperimentResult


def test_classify_dismiss_below_threshold():
    ideas = [Idea(title="Bad", description="Bad idea", weighted_score=3.0)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "dismiss"
    assert result[0].dismissed is True


def test_classify_tool_at_threshold():
    ideas = [Idea(title="Tool", description="A tool", weighted_score=4.5)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "tool"
    assert result[0].dismissed is False


def test_classify_agent_above_threshold():
    ideas = [Idea(title="Agent", description="An agent", weighted_score=5.5)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "agent"
    assert result[0].dismissed is False


def test_classify_product_above_threshold():
    ideas = [Idea(title="Product", description="A product", weighted_score=7.0)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "product"
    assert result[0].dismissed is False


def test_classify_mixed_ideas():
    ideas = [
        Idea(title="A", description="", weighted_score=3.0),  # dismiss
        Idea(title="B", description="", weighted_score=4.8),  # tool
        Idea(title="C", description="", weighted_score=5.5),  # agent
        Idea(title="D", description="", weighted_score=6.5),  # product
    ]
    result = classify_ideas(ideas)
    types = [i.artifact_type for i in result]
    assert types == ["dismiss", "tool", "agent", "product"]


def test_non_dismiss_rate_calculation():
    result = ExperimentResult(query="test")
    result.ideas_total = 10
    result.ideas_non_dismissed = 3
    result.non_dismiss_rate = 3 / 10
    assert result.non_dismiss_rate == 0.3


def test_classify_boundary_values():
    """Test exact boundary values for classification thresholds."""
    # Exactly at dismiss threshold = tool (>= 4.5)
    ideas = [Idea(title="Boundary", description="", weighted_score=4.5)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "tool"

    # Just below dismiss threshold
    ideas = [Idea(title="Below", description="", weighted_score=4.49)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "dismiss"

    # Exactly at agent threshold
    ideas = [Idea(title="Agent", description="", weighted_score=5.0)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "agent"

    # Exactly at product threshold
    ideas = [Idea(title="Product", description="", weighted_score=6.0)]
    result = classify_ideas(ideas)
    assert result[0].artifact_type == "product"
