"""Query mutator — generates variant search queries using Ollama."""

from __future__ import annotations

import logging
import random

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Context about what each agent type searches for
AGENT_CONTEXT: dict[str, str] = {
    "arxiv": (
        "arXiv academic papers. Queries search paper titles and abstracts. "
        "Good queries use technical terms like 'model context protocol', "
        "'autonomous coding agents', 'tool-augmented LLM'. "
        "Keep queries concise (2-5 words work best for arXiv)."
    ),
    "tool_monitor": (
        "GitHub repositories. Queries search repo names and descriptions. "
        "Good queries target specific technologies or frameworks: "
        "'MCP server', 'AI agent framework', 'Claude API'. "
        "Keep queries specific to avoid noisy results."
    ),
    "domain_watch": (
        "Hacker News stories. Queries search story titles and text. "
        "Good queries combine a domain + technology: "
        "'healthcare AI home health', 'solo developer AI tools'. "
        "HN search works best with natural phrases, not boolean operators."
    ),
    "youtube": (
        "YouTube videos. Queries search video titles and descriptions. "
        "Good queries target trending tech topics: "
        "'AI agents autonomous coding', 'MCP model context protocol'. "
        "Include year/recency markers for freshness."
    ),
    "rss": (
        "RSS feed articles from AI newsletters and blogs. "
        "Queries filter newsletter content. "
        "Good queries match newsletter-style topics: frameworks, tools, trends."
    ),
    "perplexity": (
        "Web search via Perplexity Sonar. Queries are full research questions. "
        "Good queries are specific, time-bounded questions: "
        "'What new AI agent frameworks launched this week?' "
        "Questions should target actionable intelligence for a solo AI developer."
    ),
    "chatgpt": (
        "Market analysis via ChatGPT. Queries are strategic analysis prompts. "
        "Good queries ask about market gaps, competitive dynamics, business models."
    ),
    "gemini_research": (
        "Google Search via Gemini grounded research. Queries are search instructions. "
        "Good queries start with 'Search for...' and target recent developments."
    ),
}


def generate_variant(
    current_query: str,
    agent: str,
    client: OllamaClient,
    all_queries: list[str] | None = None,
) -> str:
    """Generate a variant query for an agent using Ollama.

    Args:
        current_query: The query to mutate.
        agent: Agent type (determines context).
        client: OllamaClient for generation.
        all_queries: All current queries for this agent (to avoid duplicates).

    Returns:
        A new variant query string.
    """
    context = AGENT_CONTEXT.get(agent, "research signal search queries")
    existing = "\n".join(f"- {q}" for q in (all_queries or [current_query]))

    prompt = f"""You are optimizing search queries for a research signal pipeline.

Agent type: {agent}
Context: {context}

Current queries for this agent:
{existing}

The query to improve: "{current_query}"

Generate ONE alternative query that:
1. Targets the same general topic area but from a different angle
2. Is likely to surface MORE relevant, actionable signals
3. Is NOT a duplicate of any existing query
4. Follows the format guidelines for this agent type

Think about:
- What related terms or adjacent topics might surface better results?
- Is the current query too broad (noisy) or too narrow (missing signals)?
- Would a different framing catch signals the current query misses?

Respond with ONLY the new query string, nothing else. No quotes, no explanation."""

    try:
        variant = client.generate(
            prompt=prompt,
            system="You are a search query optimizer. Output ONLY the new query string.",
        ).strip()

        # Clean up: remove quotes, newlines, prefixes
        variant = variant.strip('"\'')
        variant = variant.split("\n")[0]  # Take first line only
        if variant.lower().startswith("new query:"):
            variant = variant[len("new query:"):].strip()
        if variant.lower().startswith("alternative:"):
            variant = variant[len("alternative:"):].strip()

        # Validate: not empty, not same as current, not too long
        if not variant or variant == current_query:
            logger.warning("Mutator produced invalid variant, using fallback")
            return _fallback_mutate(current_query, agent)

        if len(variant) > 200:
            variant = variant[:200]

        return variant

    except Exception as e:
        logger.warning("Mutator failed: %s, using fallback", e)
        return _fallback_mutate(current_query, agent)


def _fallback_mutate(query: str, agent: str) -> str:
    """Simple deterministic mutation when Ollama is unavailable."""
    # Strategy: append a modifier that changes the search angle
    modifiers = {
        "arxiv": [
            "benchmarks", "survey", "open-source", "multi-agent",
            "real-world applications", "scalable", "evaluation",
        ],
        "tool_monitor": [
            "typescript", "python", "rust", "API", "SDK",
            "open source", "lightweight",
        ],
        "domain_watch": [
            "startup", "enterprise", "regulation", "cost reduction",
            "emerging", "case study", "adoption",
        ],
        "youtube": [
            "tutorial", "demo", "2026", "comparison",
            "deep dive", "review",
        ],
        "perplexity": [
            "this week", "this month", "emerging", "solo developer",
            "open source", "cost-effective",
        ],
        "chatgpt": [
            "for solo consultancies", "underserved markets", "revenue models",
            "competitive moats", "emerging niches",
        ],
        "gemini_research": [
            "launched recently", "funding rounds", "open-source",
            "gaining traction", "breaking news",
        ],
    }

    agent_mods = modifiers.get(agent, ["new", "emerging", "practical"])
    mod = random.choice(agent_mods)

    # If query already contains the modifier, try another
    for _ in range(len(agent_mods)):
        if mod.lower() not in query.lower():
            break
        mod = random.choice(agent_mods)

    return f"{query} {mod}"


def select_query_to_mutate(
    queries: list[str],
    agent: str,
) -> tuple[int, str]:
    """Select which query to mutate for this experiment run.

    Strategy: round-robin by default, could be weighted by past performance later.

    Returns:
        Tuple of (query_index, query_string).
    """
    # For now, random selection. Future: weight by performance history.
    idx = random.randint(0, len(queries) - 1)
    return idx, queries[idx]
