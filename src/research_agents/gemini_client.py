"""Gemini API wrapper for YouTube transcript summarization."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from .config import (
    GEMINI_API_KEY_ENV,
    YOUTUBE_SUMMARIZER_MAX_TOKENS,
    YOUTUBE_SUMMARIZER_MODEL,
    YOUTUBE_TRANSCRIPT_MAX_CHARS,
)

logger = logging.getLogger(__name__)


def get_gemini_client() -> genai.Client:
    """Create a Gemini client from environment.

    The google-genai SDK auto-selects GOOGLE_API_KEY over any explicitly
    passed api_key. We prefer GOOGLE_API_KEY (paid tier) and fall back
    to GEMINI_API_KEY if not set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError("Missing env var GOOGLE_API_KEY or GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def summarize_transcript(
    title: str,
    transcript: str,
    client: genai.Client | None = None,
    model: str = YOUTUBE_SUMMARIZER_MODEL,
) -> dict[str, Any]:
    """Summarize a YouTube video transcript using Gemini.

    Drop-in replacement for the previous Anthropic-based summarizer.

    Returns dict with:
        summary: str - 2-3 paragraph summary of key points
        key_concepts: list[str] - main concepts/technologies discussed
        mermaid_diagram: str - Mermaid diagram of architecture/concepts
        tags: list[str] - relevant tags for categorization
    """
    if client is None:
        try:
            client = get_gemini_client()
        except RuntimeError:
            logger.warning("Gemini API key not configured — using basic transcript extraction")
            # Return a basic extraction without LLM summarization
            words = transcript.split()
            truncated = " ".join(words[:200]) + ("..." if len(words) > 200 else "")
            return {
                "summary": f"Video: {title}. Transcript excerpt: {truncated}",
                "key_concepts": [],
                "mermaid_diagram": "",
                "tags": [],
            }

    prompt = f"""Analyze this YouTube video transcript and provide a structured summary.

Video Title: {title}

Transcript (may be auto-generated, ignore minor transcription errors):
{transcript[:YOUTUBE_TRANSCRIPT_MAX_CHARS]}

Respond with JSON only:
{{
    "summary": "2-3 paragraph summary of key points and takeaways",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "mermaid_diagram": "Mermaid graph TD showing key concepts. Valid syntax.",
    "tags": ["tag1", "tag2", "tag3"]
}}

For the Mermaid diagram:
- Use `graph TD` for top-down concept maps
- Show relationships between the key technologies/concepts discussed
- Keep it to 5-10 nodes maximum
- Use descriptive edge labels

If the transcript is too short or unclear to generate meaningful output,
still provide your best summary based on the title and available content."""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=YOUTUBE_SUMMARIZER_MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip() if response.text else ""
    # Handle markdown code blocks (defensive)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini summarization response: %s", text[:200])
        result = {
            "summary": f"Video: {title}. Transcript available but summarization failed.",
            "key_concepts": [],
            "mermaid_diagram": "",
            "tags": [],
        }

    return result
