"""IdeaForge writer -- inserts synthesized ideas into IdeaForge's DB.

Uses raw SQL INSERT (no cross-project import dependency). Same decoupled
pattern as the existing Ultra-Magnus writes.

Schema targets:
  ideaforge.db -> ideas table, status='unscored'  (market signal ideas)
  ideaforge.db -> capability_gaps table            (build-failure post-mortems)
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
    signal_source TEXT DEFAULT 'unknown',
    agentic_relief TEXT,
    weight_hint TEXT
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
    struggling_user: str = "",
    weight_hint: str = "",
    agentic_relief: str = "",
    scoring_rubric: str = "",
    signal_source: str = "unknown",
    db_path: Path | None = None,
) -> int:
    """Write a synthesized idea to IdeaForge's ideas table.

    Maps idea_surfacer output to IdeaForge schema:
      title              -> title
      description        -> description
      problem_statement  -> problem_statement (Scene raw material)
      target_audience    -> target_audience
      struggling_user    -> struggling_user (first-person quote)
      weight_hint        -> weight_hint (dedicated column, R-A 1.6 2026-05-12)
      agentic_relief     -> agentic_relief (dedicated column, R-A 1.6 2026-05-12)
      scoring_rubric     -> scoring_rubric (if column exists; e.g. 'life_domain')
      source_signal_ids  -> source_signals (JSON array)
      provenance + tags  -> source_subreddits (workaround — no tags column)
      len(signal_ids)    -> signal_count
      now()              -> synthesized_at
      'unscored'         -> status

    Storage decision for weight_hint + agentic_relief (revised 2026-05-12):
      Dedicated columns. The previous "pack into score_rationale JSON
      envelope" workaround was R2 RED in the pivot drift audit — the
      life-domain scorer overwrites score_rationale with its 5-factor
      breakdown, dropping both fields before the Builder can read them.
      Dedicated columns the scorer never touches make the data-flow
      survive scoring. Falls back to the legacy score_rationale envelope
      when the migration hasn't landed (older ideaforge.db snapshots).

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

        # Legacy fallback envelope: synth-only, populated only when the
        # dedicated agentic_relief/weight_hint columns are missing. Keeps
        # older ideaforge.db snapshots from losing both fields entirely
        # before they get the R-A 1.6 migration.
        rationale_payload = {
            "rubric": scoring_rubric or "unspecified",
            "weight_hint": weight_hint,
            "agentic_relief": agentic_relief,
        }
        rationale_json = json.dumps(rationale_payload)

        # Defensive: probe the live schema. The scoring_rubric column was
        # added by an earlier migration; the agentic_relief + weight_hint
        # columns were added by R-A 1.6 (2026-05-12). Either may be absent
        # on older snapshots — fall back to the legacy column set rather
        # than crashing.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ideas)")}
        has_rubric_col = "scoring_rubric" in cols
        has_relief_cols = "agentic_relief" in cols and "weight_hint" in cols

        if has_rubric_col and has_relief_cols:
            cursor = conn.execute(
                """INSERT INTO ideas
                (title, description, problem_statement, target_audience,
                 struggling_user, score_rationale, scoring_rubric,
                 agentic_relief, weight_hint,
                 source_signals, source_subreddits, signal_count,
                 status, synthesized_at, signal_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    description,
                    problem_statement,
                    target_audience,
                    struggling_user,
                    rationale_json,
                    scoring_rubric or "life_domain",
                    agentic_relief,
                    weight_hint,
                    json.dumps(source_signal_ids),
                    json.dumps(provenance),
                    len(source_signal_ids),
                    "unscored",
                    now,
                    signal_source,
                ),
            )
        elif has_rubric_col:
            cursor = conn.execute(
                """INSERT INTO ideas
                (title, description, problem_statement, target_audience,
                 struggling_user, score_rationale, scoring_rubric,
                 source_signals, source_subreddits, signal_count,
                 status, synthesized_at, signal_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    description,
                    problem_statement,
                    target_audience,
                    struggling_user,
                    rationale_json,
                    scoring_rubric or "life_domain",
                    json.dumps(source_signal_ids),
                    json.dumps(provenance),
                    len(source_signal_ids),
                    "unscored",
                    now,
                    signal_source,
                ),
            )
        else:
            cursor = conn.execute(
                """INSERT INTO ideas
                (title, description, problem_statement, target_audience,
                 struggling_user, score_rationale,
                 source_signals, source_subreddits, signal_count,
                 status, synthesized_at, signal_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    description,
                    problem_statement,
                    target_audience,
                    struggling_user,
                    rationale_json,
                    json.dumps(source_signal_ids),
                    json.dumps(provenance),
                    len(source_signal_ids),
                    "unscored",
                    now,
                    signal_source,
                ),
            )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


# -- capability_gaps table (build-failure post-mortems) --

CAPABILITY_GAPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS capability_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    problem_statement TEXT DEFAULT '',
    target_audience TEXT DEFAULT '',
    source_signals TEXT DEFAULT '[]',
    signal_source TEXT DEFAULT 'orchestrator_reflector',
    signal_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'raw',
    created_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_capability_gaps_status
    ON capability_gaps(status);
"""


def write_capability_gap(
    title: str,
    description: str,
    source_signal_ids: list[str],
    problem_statement: str = "",
    target_audience: str = "",
    signal_source: str = "orchestrator_reflector",
    db_path: Path | None = None,
) -> int:
    """Write a capability gap to IdeaForge's capability_gaps table.

    These are internal build-failure signals from ClaudeClaw's orchestrator,
    not market ideas. They live in a separate table so they don't pollute
    the scoring/classification pipeline.

    Returns the inserted row ID.
    """
    path = db_path or _get_ideaforge_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(CAPABILITY_GAPS_SCHEMA)
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """INSERT INTO capability_gaps
            (title, description, problem_statement, target_audience,
             source_signals, signal_source, signal_count,
             status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                title,
                description,
                problem_statement,
                target_audience,
                json.dumps(source_signal_ids),
                signal_source,
                len(source_signal_ids),
                "raw",
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()
