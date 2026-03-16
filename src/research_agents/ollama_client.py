"""Ollama HTTP client for local LLM inference in the research-agents pipeline.

Replaces Anthropic API calls for relevance assessment and trend synthesis.

Default: ProBook localhost (qwen2.5:7b-instruct, CPU, always available).
Fallback: AlienPC (qwen2.5:14b, RTX 5080, faster but not always on).
Override via env: OLLAMA_BASE_URL, OLLAMA_MODEL.
"""

from __future__ import annotations

import json
import logging

import httpx

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

# AlienPC fallback config
_ALIENPC_URL = "http://10.0.0.35:11434"
_ALIENPC_MODEL = "qwen2.5:14b"

# Module-level singleton (lazy init)
_client: OllamaClient | None = None


class OllamaClient:
    """HTTP client for Ollama's REST API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a text completion."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096,
            },
        }
        if system:
            payload["system"] = system

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_json(self, system: str, prompt: str) -> dict:
        """Generate a structured JSON response.

        Uses Ollama's format="json" for reliable JSON output.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,
                "num_predict": 4096,
            },
        }

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Ollama JSON response: %s", text[:200])
            raise ValueError(f"Invalid JSON from Ollama: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is loaded."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except (httpx.HTTPError, KeyError):
            return False


def get_ollama_client() -> OllamaClient:
    """Get or create an OllamaClient, with AlienPC fallback.

    Priority:
    1. Primary (default: ProBook localhost, qwen2.5:7b-instruct)
    2. AlienPC (10.0.0.35, qwen2.5:14b) — used if primary is unavailable
    """
    global _client
    if _client is not None:
        return _client

    primary = OllamaClient()
    if primary.is_available():
        logger.info("Ollama: using %s at %s", primary.model, primary.base_url)
        _client = primary
        return _client

    # Try AlienPC fallback (skip if primary is already AlienPC)
    if primary.base_url.rstrip("/") != _ALIENPC_URL:
        fallback = OllamaClient(base_url=_ALIENPC_URL, model=_ALIENPC_MODEL)
        if fallback.is_available():
            logger.info(
                "Ollama: primary unavailable, falling back to AlienPC (%s at %s)",
                fallback.model, fallback.base_url,
            )
            _client = fallback
            return _client

    # Neither available — return primary and let calls fail with clear errors
    logger.warning(
        "Ollama: no instance available (tried %s and AlienPC). "
        "Calls will fail until Ollama is started.",
        primary.base_url,
    )
    _client = primary
    return _client


# Keywords for keyword-based fallback relevance scoring
_HIGH_KEYWORDS = {
    "mcp", "model context protocol", "claude", "anthropic", "tool use",
    "function calling", "ai agent", "autonomous coding", "hipaa",
    "home health", "agentic workflow", "mcp server",
}
_MEDIUM_KEYWORDS = {
    "llm", "ai coding", "developer tool", "prompt engineering",
    "healthcare ai", "supply chain", "ai automation", "code generation",
    "ai framework", "open source ai", "rag", "retrieval augmented",
    "agent framework", "langchain", "langgraph", "crewai",
}


def _keyword_relevance(title: str, summary: str) -> dict:
    """Simple keyword-based relevance fallback when Ollama is unavailable."""
    text = f"{title} {summary}".lower()
    high_hits = [kw for kw in _HIGH_KEYWORDS if kw in text]
    med_hits = [kw for kw in _MEDIUM_KEYWORDS if kw in text]

    if high_hits:
        relevance = "high"
        rationale = f"Matches high-priority keywords: {', '.join(high_hits[:3])}"
    elif med_hits:
        relevance = "medium"
        rationale = f"Matches medium-priority keywords: {', '.join(med_hits[:3])}"
    else:
        relevance = "low"
        rationale = "No strong keyword matches for the developer ecosystem"

    # Derive domain from matches
    domain = None
    all_hits = high_hits + med_hits
    for kw in all_hits:
        if kw in {"mcp", "model context protocol", "mcp server", "claude", "anthropic"}:
            domain = "ai-agents"
            break
        if kw in {"hipaa", "home health", "healthcare ai"}:
            domain = "healthcare-ai"
            break
        if kw in {"developer tool", "ai coding", "code generation"}:
            domain = "developer-tools"
            break
        if kw in {"supply chain"}:
            domain = "supply-chain"
            break
    if not domain and all_hits:
        domain = "ai-general"

    return {
        "relevance": relevance,
        "relevance_rationale": rationale,
        "tags": list(set(high_hits + med_hits))[:5],
        "domain": domain,
        "persona_tags": [],
    }


def assess_relevance_ollama(
    title: str,
    summary: str,
    source_context: str,
    client: OllamaClient | None = None,
) -> dict:
    """Assess signal relevance using local Ollama.

    Drop-in replacement for claude_client.assess_relevance.
    Falls back to keyword-based scoring when Ollama is unavailable.
    Returns same dict structure: relevance, relevance_rationale, tags, domain, persona_tags.
    """
    if client is None:
        client = get_ollama_client()

    # Check if Ollama is actually reachable before making the call
    if not client.is_available():
        logger.info("Ollama unavailable, using keyword-based relevance fallback")
        return _keyword_relevance(title, summary)

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
    "domain": "primary domain (e.g. ai-agents, healthcare-ai, developer-tools, etc.) or null",
    "persona_tags": ["persona_id1"]
}}"""

    try:
        result = client.generate_json(
            system="You are a research signal relevance assessor. Output valid JSON only.",
            prompt=prompt,
        )
    except (ValueError, httpx.HTTPError) as e:
        logger.warning("Ollama relevance assessment failed: %s — falling back to keywords", e)
        return _keyword_relevance(title, summary)

    return result
