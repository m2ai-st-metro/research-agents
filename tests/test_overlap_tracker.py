"""Tests for the overlap tracker (anti-monoculture monitoring)."""

from research_agents.overlap_tracker import (
    _jaccard_similarity,
    _tokenize,
)


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("MCP Server Framework v2.0")
        assert "mcp" in tokens
        assert "server" in tokens
        assert "framework" in tokens

    def test_filters_stop_words(self):
        tokens = _tokenize("A new tool for the AI developer")
        assert "new" not in tokens  # stop word
        assert "developer" in tokens

    def test_filters_short_tokens(self):
        tokens = _tokenize("AI is ok")
        assert "ok" not in tokens  # too short
        assert "ai" not in tokens  # stop word


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"mcp", "server", "framework"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        a = {"mcp", "server"}
        b = {"healthcare", "compliance"}
        assert _jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"mcp", "server", "framework"}
        b = {"mcp", "server", "tool"}
        # intersection=2, union=4
        assert _jaccard_similarity(a, b) == 0.5

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), {"mcp"}) == 0.0
        assert _jaccard_similarity(set(), set()) == 0.0
