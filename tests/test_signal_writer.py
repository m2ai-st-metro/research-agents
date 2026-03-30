"""Tests for signal_writer module."""

import sys
from pathlib import Path

import pytest

# Ensure Snow-Town contracts are importable
ST_RECORDS_ROOT = Path(__file__).resolve().parent.parent.parent / "st-records"
if str(ST_RECORDS_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_RECORDS_ROOT))

from contracts.research_signal import SignalRelevance, SignalSource
from contracts.store import ContractStore

from research_agents.signal_writer import signal_exists, write_signal


class TestWriteSignal:

    def test_write_signal_creates_record(self, store):
        signal = write_signal(
            signal_id="test-001",
            source=SignalSource.ARXIV_HF,
            title="Test Paper",
            summary="A test paper about testing",
            relevance=SignalRelevance.HIGH,
            relevance_rationale="Directly relevant",
            tags=["testing"],
            store=store,
        )
        assert signal.signal_id == "test-001"
        assert signal.source == SignalSource.ARXIV_HF

        # Verify it was written to store
        signals = store.read_signals()
        assert len(signals) == 1
        assert signals[0].title == "Test Paper"

    def test_write_signal_with_all_fields(self, store):
        signal = write_signal(
            signal_id="test-full",
            source=SignalSource.TOOL_MONITOR,
            title="New MCP Server",
            summary="A new MCP server for X",
            relevance=SignalRelevance.MEDIUM,
            relevance_rationale="Related to tooling",
            url="https://example.com",
            tags=["mcp", "tooling"],
            domain="developer-tools",
            raw_data={"stars": 100},
            store=store,
        )
        assert signal.url == "https://example.com"
        assert signal.domain == "developer-tools"
        assert signal.raw_data == {"stars": 100}


class TestSignalExists:

    def test_signal_exists_true(self, store):
        write_signal(
            signal_id="exists-001",
            source=SignalSource.ARXIV_HF,
            title="Test",
            summary="Test",
            relevance=SignalRelevance.LOW,
            store=store,
        )
        assert signal_exists("exists-001", store=store) is True

    def test_signal_exists_false(self, store):
        assert signal_exists("nonexistent", store=store) is False
