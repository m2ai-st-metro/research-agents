"""Ollama HTTP client for local LLM inference.

Drop-in replacement for Claude API calls during experiments.
Matches the interface patterns used by research-agents' claude_client
and IdeaForge's ClaudeClient.
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

MAX_RETRIES = 2
RETRY_BACKOFF = 3  # seconds, doubles each retry


class OllamaUnavailableError(Exception):
    """Raised when Ollama is unreachable or returns server errors after retries."""

logger = logging.getLogger(__name__)


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

    def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST to Ollama with retry on transient failures."""
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, MAX_RETRIES + 1, e, wait,
                    )
                    time.sleep(wait)
        raise OllamaUnavailableError(
            f"Ollama unavailable after {MAX_RETRIES + 1} attempts: {last_exc}"
        ) from last_exc

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion from Ollama.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The generated text response.

        Raises:
            OllamaUnavailableError: If Ollama is unreachable after retries.
        """
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

        resp = self._post_with_retry(payload)
        return resp.json()["response"]

    def generate_json(self, system: str, prompt: str) -> dict:
        """Generate a JSON response from Ollama.

        Matches IdeaForge's ClaudeClient.generate_json interface.

        Args:
            system: System prompt.
            prompt: User prompt.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If response is not valid JSON.
            OllamaUnavailableError: If Ollama is unreachable after retries.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,  # Lower temp for structured output
                "num_predict": 4096,
            },
        }

        resp = self._post_with_retry(payload)
        text = resp.json()["response"].strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Ollama JSON response: %s", text[:200])
            raise ValueError(f"Invalid JSON from Ollama: {e}") from e

    def assess_relevance(
        self,
        title: str,
        summary: str,
        source_context: str,
    ) -> dict:
        """Assess signal relevance — mirrors research-agents' claude_client.assess_relevance.

        Returns dict with: relevance, relevance_rationale, tags, domain
        """
        prompt = f"""Assess the relevance of this research signal to a solo AI developer's ecosystem.

The developer builds:
- Claude-powered MCP servers and tool-augmented agents
- An autonomous idea-to-product pipeline (Metroplex)
- A self-improving feedback loop (ST Metro: Metroplex -> Sky-Lynx -> ST Records)
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
    "domain": "primary domain (e.g. ai-agents, healthcare-ai, developer-tools, etc.) or null"
}}"""

        try:
            result = self.generate_json(
                system="You are a research signal relevance assessor. Output valid JSON only.",
                prompt=prompt,
            )
        except (ValueError, OllamaUnavailableError) as e:
            logger.warning("Relevance assessment failed: %s", e)
            result = {
                "relevance": "low",
                "relevance_rationale": f"Assessment unavailable: {type(e).__name__}",
                "tags": [],
                "domain": None,
            }

        return result

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is loaded."""
        try:
            resp = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model is available (handle tag variations)
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except (httpx.HTTPError, KeyError):
            return False
