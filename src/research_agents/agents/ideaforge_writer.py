"""IdeaForge writer -- inserts synthesized ideas into IdeaForge's ideas table.

Uses raw SQL INSERT (no cross-project import dependency). Same decoupled
pattern as the existing Ultra-Magnus writes.

Schema target: ideaforge.db -> ideas table, status='unscored'
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import IDEAFORGE_DB

logger = logging.getLogger(__name__)

# IdeaForge ideas table DDL (bootstrap if DB exists but table doesn't)
IDEAFORGE_IDEAS_SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    problem_statement TEXT DEFAULT '',
    target_audience TEXT DEFAULT '',
    source_signals TEXT DEFAULT '[]',
    source_subreddits TEXT DEFAULT '[]',
    signal_count INTEGER DEFAULT 0,
    opportunity_score REAL,
    problem_score REAL,
    feasibility_score REAL,
    why_now_score REAL,
    competition_score REAL,
    weighted_score REAL,
    score_rationale TEXT,
    artifact_type TEXT,
    route_rationale TEXT,
    route_confidence REAL,
    struggling_user TEXT,
    classified_at TIMESTAMP,
    status TEXT DEFAULT 'unscored',
    synthesized_at TIMESTAMP,
    scored_at TIMESTAMP,
    exported_at TIMESTAMP,
    ultra_magnus_id INTEGER,
    signal_source TEXT DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS idx_ideas_weighted_score ON ideas(weighted_score);
"""


def _get_ideaforge_db_path() -> Path:
    """Get IdeaForge DB path from env or config default."""
    env_path = os.environ.get("IDEAFORGE_DB")
    if env_path:
        return Path(env_path)
    return IDEAFORGE_DB


def write_idea_to_ideaforge(
    title: str,
    description: str,
    tags: list[str],
    source_signal_ids: list[str],
    problem_statement: str = "",
    target_audience: str = "",
    signal_source: str = "unknown",
    db_path: Path | None = None,
) -> int:
    """Write a synthesized idea to IdeaForge's ideas table.

    Maps idea_surfacer output to IdeaForge schema:
      title              -> title
      description        -> description
      problem_statement  -> problem_statement
      target_audience    -> target_audience
      source_signal_ids  -> source_signals (JSON array)
      provenance tag     -> source_subreddits[0] (workaround for provenance)
      len(signal_ids)    -> signal_count
      now()              -> synthesized_at
      'unscored'         -> status

    Returns the inserted idea row ID.
    """
    path = db_path or _get_ideaforge_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        # Ensure table exists (idempotent)
        conn.executescript(IDEAFORGE_IDEAS_SCHEMA)

        now = datetime.now(timezone.utc).isoformat()

        # Store provenance + tags in source_subreddits as workaround
        # (IdeaForge schema has no dedicated tags column)
        provenance = ["research-agents:idea-surfacer"] + tags

        cursor = conn.execute(
            """INSERT INTO ideas
            (title, description, problem_statement, target_audience,
             source_signals, source_subreddits, signal_count,
             signal_source, status, synthesized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                title,
                description,
                problem_statement,
                target_audience,
                json.dumps(source_signal_ids),
                json.dumps(provenance),
                len(source_signal_ids),
                signal_source,
                "unscored",
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()
