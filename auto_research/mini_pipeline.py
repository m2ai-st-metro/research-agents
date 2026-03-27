"""Mini-pipeline — self-contained experiment chain.

Runs the full signal-to-classification chain using local Ollama inference.
Does NOT write to production databases — everything stays in memory for evaluation.

Chain: query → agent search (free APIs) → Ollama relevance → Ollama synthesis
       → Ollama scoring → classification (threshold-based) → metrics
"""

from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import feedparser  # type: ignore[import-untyped]
import httpx

from .config import (
    DISMISS_THRESHOLD,
    ROUTE_THRESHOLDS,
    SCORE_WEIGHTS,
)
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
GITHUB_API_URL = "https://api.github.com/search/repositories"
HN_API_URL = "https://hn.algolia.com/api/v1/search_by_date"


@dataclass
class Signal:
    """A research signal discovered during experiment."""
    signal_id: str
    title: str
    summary: str
    source: str
    url: str
    relevance: str = "low"
    relevance_rationale: str = ""
    tags: list[str] = field(default_factory=list)
    domain: str | None = None


@dataclass
class Idea:
    """A synthesized idea from signals."""
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    source_signal_ids: list[str] = field(default_factory=list)
    opportunity: float = 0.0
    problem: float = 0.0
    feasibility: float = 0.0
    why_now: float = 0.0
    competition: float = 0.0
    weighted_score: float = 0.0
    artifact_type: str = "dismiss"
    dismissed: bool = True


@dataclass
class ExperimentResult:
    """Result of running a single experiment (baseline or variant)."""
    query: str
    signals_found: int = 0
    signals_relevant: int = 0
    ideas_synthesized: int = 0
    ideas_non_dismissed: int = 0
    ideas_total: int = 0
    non_dismiss_rate: float = 0.0
    avg_weighted_score: float = 0.0
    signals: list[Signal] = field(default_factory=list)
    ideas: list[Idea] = field(default_factory=list)


# --- Source Searchers ---

def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv API. Returns paper metadata."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        resp = httpx.get(ARXIV_API_URL, params=params, timeout=30.0, follow_redirects=True)
        if resp.status_code == 429:
            time.sleep(5)
            resp = httpx.get(ARXIV_API_URL, params=params, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("arXiv API error for '%s': %s", query, e)
        return []

    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id_raw = entry.findtext("atom:id", "", ARXIV_NS)
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
        title = entry.findtext("atom:title", "", ARXIV_NS).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", ARXIV_NS).strip().replace("\n", " ")

        papers.append({
            "signal_id": f"arxiv-{arxiv_id.replace('/', '-')}",
            "title": title,
            "summary": summary[:500],
            "url": arxiv_id_raw,
            "source": "arxiv",
        })
    return papers


def search_github(query: str, max_results: int = 10) -> list[dict]:
    """Search GitHub repositories. Returns repo metadata."""
    params = {
        "q": f"{query} pushed:>2026-01-01",
        "sort": "updated",
        "per_page": max_results,
    }
    try:
        resp = httpx.get(GITHUB_API_URL, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("GitHub API error for '%s': %s", query, e)
        return []

    items = resp.json().get("items", [])
    repos = []
    for item in items:
        desc = item.get("description") or ""
        repos.append({
            "signal_id": f"tool-{hashlib.sha256(item['html_url'].encode()).hexdigest()[:12]}",
            "title": item["full_name"],
            "summary": desc[:500],
            "url": item["html_url"],
            "source": "github",
        })
    return repos


def search_hn(query: str, max_results: int = 10) -> list[dict]:
    """Search Hacker News via Algolia. Returns story metadata."""
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": max_results,
    }
    try:
        resp = httpx.get(HN_API_URL, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("HN API error for '%s': %s", query, e)
        return []

    hits = resp.json().get("hits", [])
    stories = []
    for hit in hits:
        title = hit.get("title") or ""
        # HN stories may have no body text
        text = hit.get("story_text") or hit.get("comment_text") or ""
        stories.append({
            "signal_id": f"hn-{hit.get('objectID', '')}",
            "title": title,
            "summary": text[:500] if text else title,
            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            "source": "hn",
        })
    return stories


def search_youtube(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube by scraping public search results (no API key needed).

    Parses ytInitialData from the search results HTML to extract video metadata.
    """
    import json as _json
    import re as _re

    try:
        resp = httpx.get(
            "https://www.youtube.com/results",
            params={"search_query": query},
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("YouTube search failed for '%s': %s", query, e)
        return []

    match = _re.search(r"var ytInitialData = ({.*?});</script>", resp.text)
    if not match:
        logger.warning("Could not extract ytInitialData for '%s'", query)
        return []

    try:
        data = _json.loads(match.group(1))
        contents = (
            data["contents"]["twoColumnSearchResultsRenderer"]
            ["primaryContents"]["sectionListRenderer"]["contents"][0]
            ["itemSectionRenderer"]["contents"]
        )
    except (KeyError, IndexError, _json.JSONDecodeError) as e:
        logger.warning("Failed to parse YouTube search data: %s", e)
        return []

    videos = []
    for item in contents:
        vid = item.get("videoRenderer")
        if not vid:
            continue
        video_id = vid.get("videoId", "")
        title = vid.get("title", {}).get("runs", [{}])[0].get("text", "")
        desc = ""
        if "detailedMetadataSnippets" in vid:
            desc = "".join(
                r.get("text", "") for r in
                vid["detailedMetadataSnippets"][0].get("snippetText", {}).get("runs", [])
            )
        elif "descriptionSnippet" in vid:
            desc = "".join(
                r.get("text", "") for r in vid["descriptionSnippet"].get("runs", [])
            )
        videos.append({
            "signal_id": f"youtube-{video_id}",
            "title": title,
            "summary": (desc or title)[:500],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "source": "youtube",
        })
        if len(videos) >= max_results:
            break

    return videos


def search_rss(query: str, max_results: int = 10) -> list[dict]:
    """Search across configured RSS feeds, filtering entries by query relevance.

    Fetches all feeds from the production config, then filters entries
    whose title or summary contains query terms (case-insensitive).
    """
    import sys
    from pathlib import Path

    # Load production RSS_FEEDS config
    config_path = Path(__file__).resolve().parent.parent / "src" / "research_agents"
    if str(config_path.parent) not in sys.path:
        sys.path.insert(0, str(config_path.parent))

    try:
        import importlib
        import research_agents.config as ra_config
        importlib.reload(ra_config)
        feeds = getattr(ra_config, "RSS_FEEDS", [])
    except ImportError:
        logger.warning("Cannot import RSS_FEEDS from research_agents.config")
        return []

    # Tokenize query for matching
    query_terms = query.lower().split()
    articles = []

    for feed_cfg in feeds:
        if feed_cfg.get("parser") != "feedparser":
            continue  # Skip firecrawl-only feeds in experiments
        try:
            result = feedparser.parse(feed_cfg["url"])
        except Exception as e:
            logger.debug("Feed %s failed: %s", feed_cfg["name"], e)
            continue

        for entry in result.entries:
            title = entry.get("title", "")
            summary = str(entry.get("summary", entry.get("description", "")))[:500]
            text = f"{title} {summary}".lower()

            # Match if any query term appears in title/summary
            if any(term in text for term in query_terms):
                url = entry.get("link", "")
                articles.append({
                    "signal_id": f"rss-{hashlib.sha256(url.encode()).hexdigest()[:12]}",
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "source": f"rss-{feed_cfg['name']}",
                })

        if len(articles) >= max_results:
            break

    return articles[:max_results]


# --- Paid Agent Searchers ---
# These call external LLM APIs that return pre-structured signals.
# Queries are full research prompts, not keyword searches.
# Cost: ~$0.01-0.10 per query depending on model.

PERPLEXITY_API_URL_PAID = "https://api.perplexity.ai/chat/completions"
OPENAI_API_URL_PAID = "https://api.openai.com/v1/chat/completions"

_PAID_SYSTEM_PROMPT = """You are a research analyst for a solo AI developer/consultant.
Identify 2-5 actionable signals: tools, market shifts, regulatory changes, or opportunities.

Focus: AI agents, MCP servers, healthcare AI (HIPAA), developer tools, workflow automation.

Return JSON only:
{"signals": [{"title": "...", "summary": "2-3 sentences", "url": "source URL or null", "relevance": "high|medium|low", "tags": ["tag1"], "domain": "ai-agents|healthcare-ai|developer-tools|etc"}]}"""


def _parse_paid_signals(result: dict, source_prefix: str) -> list[dict]:
    """Convert paid agent JSON response to standard signal format."""
    signals = []
    for s in result.get("signals", []):
        title = s.get("title", "")
        sig_id = f"{source_prefix}-{hashlib.sha256(f'{title}'.encode()).hexdigest()[:12]}"
        signals.append({
            "signal_id": sig_id,
            "title": title,
            "summary": s.get("summary", title)[:500],
            "url": s.get("url") or "",
            "source": source_prefix,
        })
    return signals


def search_perplexity(query: str, max_results: int = 10) -> list[dict]:
    """Search via Perplexity Sonar API. Costs ~$0.01/query."""
    import os
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.warning("PERPLEXITY_API_KEY not set, skipping perplexity search")
        return []

    try:
        resp = httpx.post(
            PERPLEXITY_API_URL_PAID,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": _PAID_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 4096,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        import json as _json
        result = _json.loads(text)
    except Exception as e:
        logger.warning("Perplexity search failed for '%s': %s", query[:60], e)
        return []

    return _parse_paid_signals(result, "perplexity")[:max_results]


def search_chatgpt(query: str, max_results: int = 10) -> list[dict]:
    """Search via OpenAI ChatGPT API. Costs ~$0.02-0.05/query."""
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping chatgpt search")
        return []

    # Read model from research-agents config to avoid hardcoding
    import sys
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "src" / "research_agents"
    if str(config_path.parent) not in sys.path:
        sys.path.insert(0, str(config_path.parent))
    try:
        import importlib
        import research_agents.config as ra_config
        importlib.reload(ra_config)
        model = getattr(ra_config, "CHATGPT_MODEL", "gpt-4.1")
    except ImportError:
        model = "gpt-4.1"

    try:
        resp = httpx.post(
            OPENAI_API_URL_PAID,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _PAID_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "max_completion_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        import json as _json
        result = _json.loads(text)
    except Exception as e:
        logger.warning("ChatGPT search failed for '%s': %s", query[:60], e)
        return []

    return _parse_paid_signals(result, "chatgpt")[:max_results]


def search_gemini(query: str, max_results: int = 10) -> list[dict]:
    """Search via Gemini with Google Search grounding. Costs ~$0.01/query."""
    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping gemini search")
        return []

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Read model from research-agents config
        import sys
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "src" / "research_agents"
        if str(config_path.parent) not in sys.path:
            sys.path.insert(0, str(config_path.parent))
        try:
            import importlib
            import research_agents.config as ra_config
            importlib.reload(ra_config)
            model = getattr(ra_config, "GEMINI_RESEARCH_MODEL", "gemini-2.0-flash")
        except ImportError:
            model = "gemini-2.0-flash"

        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        response = client.models.generate_content(
            model=model,
            contents=f"{_PAID_SYSTEM_PROMPT}\n\nResearch query: {query}",
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                tools=[google_search_tool],
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip() if response.text else ""
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        import json as _json
        result = _json.loads(text)
    except Exception as e:
        logger.warning("Gemini search failed for '%s': %s", query[:60], e)
        return []

    return _parse_paid_signals(result, "gemini")[:max_results]


# Maps agent name to search function
AGENT_SEARCHERS: dict[str, callable] = {
    "arxiv": search_arxiv,
    "tool_monitor": search_github,
    "domain_watch": search_hn,
    "youtube": search_youtube,
    "rss": search_rss,
    "perplexity": search_perplexity,
    "chatgpt": search_chatgpt,
    "gemini_research": search_gemini,
}


# --- Pipeline Stages ---

def assess_signals(
    raw_signals: list[dict],
    client: OllamaClient,
    min_relevance: str = "medium",
) -> list[Signal]:
    """Run Ollama relevance assessment on raw signals. Returns filtered signals."""
    relevance_order = {"high": 3, "medium": 2, "low": 1}
    min_level = relevance_order.get(min_relevance, 2)

    assessed = []
    for raw in raw_signals:
        assessment = client.assess_relevance(
            title=raw["title"],
            summary=raw["summary"],
            source_context=f"{raw['source']} signal",
        )

        relevance = assessment.get("relevance", "low")
        if relevance_order.get(relevance, 0) >= min_level:
            assessed.append(Signal(
                signal_id=raw["signal_id"],
                title=raw["title"],
                summary=raw["summary"],
                source=raw["source"],
                url=raw["url"],
                relevance=relevance,
                relevance_rationale=assessment.get("relevance_rationale", ""),
                tags=assessment.get("tags", []),
                domain=assessment.get("domain"),
            ))

    return assessed


def synthesize_ideas(signals: list[Signal], client: OllamaClient) -> list[Idea]:
    """Use Ollama to synthesize signals into ideas."""
    if not signals:
        return []

    signal_summaries = []
    for s in signals:
        signal_summaries.append(
            f"- [{s.source}] {s.title}: {s.summary} "
            f"(relevance: {s.relevance}, domain: {s.domain or 'general'})"
        )

    prompt = f"""You are a project idea synthesizer for a solo AI developer. Given these research signals, identify 0-3 actionable project ideas.

The developer's ecosystem:
- Claude-powered MCP servers and tool-augmented agents
- An autonomous idea-to-product pipeline (Ultra Magnus)
- A self-improving feedback loop (Snow-Town)
- Healthcare AI projects (HIPAA-compliant, home health focus)
- Developer productivity tools

Research signals:
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
- If no clear ideas emerge, return {{"ideas": []}}
- Maximum 3 ideas per synthesis run"""

    try:
        result = client.generate_json(
            system="You are a project idea synthesizer. Output valid JSON only.",
            prompt=prompt,
        )
        raw_ideas = result.get("ideas", [])
    except ValueError:
        logger.warning("Failed to synthesize ideas from %d signals", len(signals))
        return []

    ideas = []
    for raw in raw_ideas:
        ideas.append(Idea(
            title=raw.get("title", "Untitled"),
            description=raw.get("description", ""),
            tags=raw.get("tags", []),
            source_signal_ids=raw.get("source_signal_ids", []),
        ))
    return ideas


def score_ideas(ideas: list[Idea], client: OllamaClient) -> list[Idea]:
    """Score ideas on 5 dimensions using Ollama."""
    scoring_system = """You are an expert startup evaluator. You score product ideas on 5 dimensions, each from 0-10.

Scoring criteria:
1. **Opportunity** (0-10): Market size, revenue potential, willingness to pay
2. **Problem** (0-10): Pain severity, frequency, current workaround quality
3. **Feasibility** (0-10): Can a solo developer build an MVP in 2-4 weeks? Tech complexity, data availability
4. **Why Now** (0-10): Timing — new regulations, tech shifts, market trends making this timely
5. **Competition** (0-10): Competitive landscape favorability (10 = no competition, 0 = dominated by incumbents)

Be calibrated:
- 7+ is genuinely good
- 5 is average
- 3 or below means significant concerns
- Don't be afraid to give low scores

Output ONLY valid JSON, no markdown or explanation."""

    for idea in ideas:
        prompt = f"""Score this product idea:

TITLE: {idea.title}
DESCRIPTION: {idea.description}

Output JSON:
{{
    "opportunity": <0-10>,
    "problem": <0-10>,
    "feasibility": <0-10>,
    "why_now": <0-10>,
    "competition": <0-10>
}}"""

        try:
            result = client.generate_json(scoring_system, prompt)
            idea.opportunity = float(result.get("opportunity", 0))
            idea.problem = float(result.get("problem", 0))
            idea.feasibility = float(result.get("feasibility", 0))
            idea.why_now = float(result.get("why_now", 0))
            idea.competition = float(result.get("competition", 0))

            idea.weighted_score = round(
                idea.opportunity * SCORE_WEIGHTS["opportunity"]
                + idea.problem * SCORE_WEIGHTS["problem"]
                + idea.feasibility * SCORE_WEIGHTS["feasibility"]
                + idea.why_now * SCORE_WEIGHTS["why_now"]
                + idea.competition * SCORE_WEIGHTS["competition"],
                2,
            )
        except (ValueError, TypeError) as e:
            logger.warning("Scoring failed for '%s': %s", idea.title, e)

    return ideas


def classify_ideas(ideas: list[Idea]) -> list[Idea]:
    """Classify ideas using threshold-based logic (no LLM needed).

    This mirrors IdeaForge's _apply_thresholds — the threshold guardrails
    override Claude's classification anyway, so we can skip the LLM call
    and use the thresholds directly for experiment purposes.
    """
    for idea in ideas:
        ws = idea.weighted_score

        if ws < DISMISS_THRESHOLD:
            idea.artifact_type = "dismiss"
            idea.dismissed = True
        elif ws >= ROUTE_THRESHOLDS["product"]:
            idea.artifact_type = "product"
            idea.dismissed = False
        elif ws >= ROUTE_THRESHOLDS["agent"]:
            idea.artifact_type = "agent"
            idea.dismissed = False
        elif ws >= ROUTE_THRESHOLDS["tool"]:
            idea.artifact_type = "tool"
            idea.dismissed = False
        else:
            idea.artifact_type = "dismiss"
            idea.dismissed = True

    return ideas


# --- Full Pipeline ---

def run_experiment(
    query: str,
    agent: str,
    client: OllamaClient,
    max_results: int = 10,
    min_relevance: str = "low",
) -> ExperimentResult:
    """Run the full mini-pipeline for a single query.

    Args:
        query: Search query to test.
        agent: Agent type (determines which API to search).
        client: OllamaClient for inference.
        max_results: Max results per search.
        min_relevance: Minimum relevance threshold.

    Returns:
        ExperimentResult with all metrics.
    """
    result = ExperimentResult(query=query)

    # Step 1: Search
    searcher = AGENT_SEARCHERS.get(agent)
    if searcher is None:
        logger.warning("No searcher for agent '%s', skipping", agent)
        result.signals_found = 0
        return result

    raw_signals = searcher(query, max_results=max_results)
    result.signals_found = len(raw_signals)

    if not raw_signals:
        return result

    # Step 2: Relevance assessment
    signals = assess_signals(raw_signals, client, min_relevance=min_relevance)
    result.signals_relevant = len(signals)
    result.signals = signals

    if not signals:
        return result

    # Step 3: Idea synthesis
    ideas = synthesize_ideas(signals, client)
    result.ideas_synthesized = len(ideas)

    if not ideas:
        return result

    # Step 4: Scoring
    ideas = score_ideas(ideas, client)

    # Step 5: Classification (threshold-based)
    ideas = classify_ideas(ideas)
    result.ideas = ideas
    result.ideas_total = len(ideas)
    result.ideas_non_dismissed = sum(1 for i in ideas if not i.dismissed)

    # Compute metrics
    if ideas:
        result.non_dismiss_rate = result.ideas_non_dismissed / result.ideas_total
        scores = [i.weighted_score for i in ideas if i.weighted_score > 0]
        result.avg_weighted_score = sum(scores) / len(scores) if scores else 0.0

    return result
