"""Signal writer: writes ResearchSignal records to Snow-Town ContractStore."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import SNOW_TOWN_ROOT

logger = logging.getLogger(__name__)

# Add Snow-Town to sys.path so we can import contracts
_snow_town_path = str(SNOW_TOWN_ROOT)
if _snow_town_path not in sys.path:
    sys.path.insert(0, _snow_town_path)

from contracts import ContractStore  # noqa: E402
from contracts.research_signal import ResearchSignal, SignalRelevance, SignalSource  # noqa: E402


def get_store(db_path: Path | None = None) -> ContractStore:
    """Get a ContractStore instance pointing to Snow-Town data."""
    data_dir = SNOW_TOWN_ROOT / "data"
    return ContractStore(data_dir=data_dir, db_path=db_path)


def write_signal(
    signal_id: str,
    source: SignalSource,
    title: str,
    summary: str,
    relevance: SignalRelevance,
    relevance_rationale: str = "",
    url: str | None = None,
    tags: list[str] | None = None,
    domain: str | None = None,
    raw_data: dict | None = None,
    store: ContractStore | None = None,
) -> ResearchSignal:
    """Create and write a ResearchSignal to the ContractStore.

    Returns the written signal.
    """
    signal = ResearchSignal(
        signal_id=signal_id,
        source=source,
        title=title,
        summary=summary,
        url=url,
        relevance=relevance,
        relevance_rationale=relevance_rationale,
        tags=tags or [],
        domain=domain,
        raw_data=raw_data,
    )

    if store is None:
        store = get_store()
    try:
        store.write_signal(signal)
        logger.info(f"Wrote signal {signal_id}: {title}")
    finally:
        if store is not None:
            store.close()

    return signal


def signal_exists(signal_id: str, store: ContractStore | None = None) -> bool:
    """Check if a signal with this ID already exists."""
    close_after = store is None
    if store is None:
        store = get_store()
    try:
        results = store.query_signals(limit=1)
        # Check via direct SQLite query for efficiency
        conn = store._get_conn()
        row = conn.execute(
            "SELECT 1 FROM research_signals WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        return row is not None
    finally:
        if close_after:
            store.close()
