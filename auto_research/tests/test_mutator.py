"""Tests for the query mutator (fallback logic only — Ollama tests need a running server)."""

from auto_research.mutator import _fallback_mutate, select_query_to_mutate


def test_fallback_mutate_appends_modifier():
    result = _fallback_mutate("AI coding assistant", "tool_monitor")
    assert result.startswith("AI coding assistant ")
    assert len(result) > len("AI coding assistant ")


def test_fallback_mutate_different_agents():
    arxiv = _fallback_mutate("MCP protocol", "arxiv")
    github = _fallback_mutate("MCP protocol", "tool_monitor")
    hn = _fallback_mutate("MCP protocol", "domain_watch")

    # All should be different from original
    assert arxiv != "MCP protocol"
    assert github != "MCP protocol"
    assert hn != "MCP protocol"


def test_fallback_mutate_unknown_agent():
    result = _fallback_mutate("test query", "unknown_agent")
    assert result.startswith("test query ")


def test_select_query_returns_valid_index():
    queries = ["q1", "q2", "q3"]
    idx, query = select_query_to_mutate(queries, "arxiv")
    assert 0 <= idx < len(queries)
    assert query == queries[idx]


def test_select_query_single_item():
    queries = ["only_one"]
    idx, query = select_query_to_mutate(queries, "arxiv")
    assert idx == 0
    assert query == "only_one"
