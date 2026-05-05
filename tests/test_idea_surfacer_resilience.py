"""Resilience tests for idea_surfacer JSON parser and provider fallback.

Three failure modes covered:
  T1 - Truncated JSON detection (open brace, never closes)
  T2 - Stray-brace-in-prose adversarial extraction
  T3 - Provider fallback after primary retry exhaustion

These tests are RED against unmodified idea_surfacer.py. They drive the
builder-phase fix that adds a typed TruncatedJSONError, multi-candidate
brace scanning, and an IDEA_SURFACER_FALLBACK_MODEL env-driven hop.

Mock-only -- no live API calls. Model identifiers are abstract sentinels.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from contracts.research_signal import ResearchSignal, SignalRelevance, SignalSource

from research_agents.agents import idea_surfacer
from research_agents.agents.idea_surfacer import (
    _synthesize_ideas,
    _try_parse_ideas_json,
)


def _signal(signal_id: str, source: SignalSource) -> ResearchSignal:
    return ResearchSignal(
        signal_id=signal_id,
        source=source,
        title=f"Title {signal_id}",
        summary=f"Summary {signal_id}",
        relevance=SignalRelevance.HIGH,
        emitted_at=datetime.now(),
    )


@pytest.fixture
def diverse_signals() -> list[ResearchSignal]:
    return [
        _signal("sig-1", SignalSource.ARXIV_HF),
        _signal("sig-2", SignalSource.RSS_SCANNER),
        _signal("sig-3", SignalSource.REDDIT),
    ]


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# T1 - Truncated JSON detection
# ---------------------------------------------------------------------------

class TestTruncatedJSON:

    TRUNCATED_INPUT = (
        '{"ideas":[{"title":"x","description":"truncated mid-string '
        "and the response was cut off here"
    )

    def test_truncated_json_logs_specific_truncation_message(self, caplog):
        caplog.set_level(logging.WARNING, logger="research_agents.agents.idea_surfacer")

        _try_parse_ideas_json(self.TRUNCATED_INPUT)

        all_messages = " ".join(rec.getMessage() for rec in caplog.records)
        assert "No JSON object found" not in all_messages, (
            "Truncated input must not be reported as 'No JSON object found' - "
            "the opening brace was present, the JSON was just cut off."
        )
        assert any(
            tok in all_messages.lower()
            for tok in ("truncat", "unbalanced", "incomplete", "unterminated")
        ), (
            "Expected a truncation-specific warning. Got: "
            + (all_messages or "<no log records>")
        )

    def test_truncated_json_raises_or_salvages(self):
        salvage_input = (
            '{"ideas":['
            '{"title":"Complete Idea","description":"Fully closed inner object"},'
            '{"title":"Truncated","description":"this one is cut'
        )

        TruncatedJSONError = getattr(idea_surfacer, "TruncatedJSONError", None)
        assert TruncatedJSONError is not None, (
            "Expected idea_surfacer.TruncatedJSONError to be defined as a typed "
            "exception so callers can distinguish truncation from no-JSON-at-all."
        )

        try:
            result = _try_parse_ideas_json(salvage_input)
        except TruncatedJSONError:
            return

        assert result is not None and len(result) >= 1, (
            "Parser returned None for input that clearly started a JSON object. "
            "Either raise TruncatedJSONError or salvage the complete inner idea."
        )
        assert result[0].get("title") == "Complete Idea"


# ---------------------------------------------------------------------------
# T2 - Stray-brace-in-prose adversarial extraction
# ---------------------------------------------------------------------------

class TestStrayBraceInProse:

    def test_stray_brace_in_prose_then_real_json(self):
        text = (
            "We need to extract JSON. The schema looks like "
            "{key: value, nested: {inner: thing}}. "
            'Here is the answer: {"ideas":[{"title":"Real Idea","description":"Real desc"}]}'
        )
        parsed = _try_parse_ideas_json(text)
        assert parsed is not None, (
            "Stray-brace prose followed by real JSON should still parse the real "
            "JSON. Today, the first-brace-wins extractor locks onto the prose "
            "snippet and discards the real payload."
        )
        assert parsed == [{"title": "Real Idea", "description": "Real desc"}]


# ---------------------------------------------------------------------------
# T3 - Provider fallback after primary retry exhaustion
# ---------------------------------------------------------------------------

class TestProviderFallback:

    PRIMARY_SENTINEL = "primary-model-sentinel"
    FALLBACK_SENTINEL = "fallback-model-sentinel"

    def test_fallback_called_after_primary_three_failures(
        self, diverse_signals, monkeypatch
    ):
        monkeypatch.setattr(
            idea_surfacer, "CLAUDE_MODEL", self.PRIMARY_SENTINEL, raising=True
        )
        monkeypatch.setattr(
            idea_surfacer,
            "IDEA_SURFACER_FALLBACK_MODEL",
            self.FALLBACK_SENTINEL,
            raising=False,
        )

        primary_garbage = "definitely not json - this is unstructured prose"
        fallback_valid = (
            '{"ideas":[{"title":"Fallback Idea","description":"From fallback model"}]}'
        )

        def create_side_effect(*, model, **kwargs):
            if model == self.PRIMARY_SENTINEL:
                return _mock_response(primary_garbage)
            if model == self.FALLBACK_SENTINEL:
                return _mock_response(fallback_valid)
            raise AssertionError(f"Unexpected model id: {model!r}")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = create_side_effect

        with patch(
            "research_agents.agents.idea_surfacer.get_client",
            return_value=mock_client,
        ):
            ideas = _synthesize_ideas(diverse_signals, dry_run=False)

        assert ideas == [
            {"title": "Fallback Idea", "description": "From fallback model"}
        ], (
            "Fallback model returned valid JSON but surfacer didn't use it. "
            "Expected one idea from the fallback hop; got: " + repr(ideas)
        )

        all_calls = mock_client.chat.completions.create.call_args_list
        primary_calls = [c for c in all_calls if c.kwargs.get("model") == self.PRIMARY_SENTINEL]
        fallback_calls = [c for c in all_calls if c.kwargs.get("model") == self.FALLBACK_SENTINEL]

        assert len(primary_calls) == 3, (
            f"Expected exactly 3 primary-model calls before fallback; "
            f"got {len(primary_calls)}"
        )
        assert len(fallback_calls) == 1, (
            f"Expected exactly 1 fallback-model call after primary exhaustion; "
            f"got {len(fallback_calls)}"
        )
