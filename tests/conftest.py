"""Shared fixtures for research-agents tests."""

import sys
from pathlib import Path

import pytest

# Ensure Snow-Town contracts are importable in tests
SNOW_TOWN_ROOT = Path(__file__).resolve().parent.parent.parent / "snow-town"
if str(SNOW_TOWN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNOW_TOWN_ROOT))

from contracts.store import ContractStore


@pytest.fixture
def store(tmp_path):
    """Create a ContractStore with temp directory."""
    s = ContractStore(data_dir=tmp_path, db_path=tmp_path / "test.db")
    yield s
    s.close()
