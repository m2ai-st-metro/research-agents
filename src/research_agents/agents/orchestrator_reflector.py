"""Orchestrator Reflector Agent.

Reads post-mortem data from the agentic-orchestrator SQLite database
(claudeclaw/store/orchestrator.db) and turns escalations, replans, and
low-composite-score outcomes into capability gaps in IdeaForge's
capability_gaps table (status='raw', signal_source='orchestrator_reflector').

Changed 2026-04-15: writes to capability_gaps instead of ideas table
to stop internal build-failure signals from polluting the scoring pipeline.

Deterministic templating -- no LLM call. This is intentional: the existing
LLM synthesis path (idea_surfacer) is known to hit Nemotron-3 JSON bugs,
and templating keeps this loop cheap, fast, and testable. Upgrade to LLM
synthesis only if the templated ideas score consistently low in IdeaForge.

Cursor state lives at claudeclaw/store/orchestrator_reflector_state.json
so repeat runs only process new rows.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ideaforge_writer import write_capability_gap

logger = logging.getLogger(__name__)


ORCHESTRATOR_DB = Path(os.environ.get(
    "ORCHESTRATOR_DB",
    str(Path.home() / "projects" / "claudeclaw" / "store" / "orchestrator.db"),
))

CURSOR_PATH = Path(os.environ.get(
    "ORCHESTRATOR_REFLECTOR_CURSOR",
    str(Path.home() / "projects" / "claudeclaw" / "store" / "orchestrator_reflector_state.json"),
))

# Threshold: outcomes at or below this composite score are treated as failures
# worth reflecting on. 0.5 is the midpoint of the 0-1 judge scale and below
# ClaudeClaw's 0.45 acceptThreshold default -- roughly "the judge thought this
# answer was more wrong than right."
COMPOSITE_FAILURE_THRESHOLD = 0.5

# Decision types that indicate the orchestrator itself detected a problem
# worth surfacing as a capability-gap signal.
REFLECTABLE_DECISION_TYPES = frozenset({"escalate", "replan", "reassign"})


@dataclass
class ReflectorCursor:
    """Tracks the last processed row IDs so repeat runs are idempotent."""

    last_outcome_id: int = 0
    last_decision_id: int = 0

    @classmethod
    def load(cls, path: Path) -> ReflectorCursor:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                last_outcome_id=int(data.get("last_outcome_id", 0)),
                last_decision_id=int(data.get("last_decision_id", 0)),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Cursor file %s is corrupt (%s) -- starting from 0", path, e)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "last_outcome_id": self.last_outcome_id,
            "last_decision_id": self.last_decision_id,
        }, indent=2))


def _truncate(text: str | None, limit: int = 400) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _fetch_failing_outcomes(
    conn: sqlite3.Connection, after_id: int, threshold: float
) -> list[sqlite3.Row]:
    """Outcomes strictly newer than cursor whose composite is at or below threshold."""
    rows = conn.execute(
        """
        SELECT id, mission_id, subtask_id, task_type, agent_id, status,
               composite_score, judge_reasoning, judge_method, created_at
        FROM outcome_logs
        WHERE id > ? AND composite_score IS NOT NULL AND composite_score <= ?
        ORDER BY id ASC
        """,
        (after_id, threshold),
    ).fetchall()
    return rows


def _fetch_reflectable_decisions(
    conn: sqlite3.Connection, after_id: int
) -> list[sqlite3.Row]:
    """Decisions strictly newer than cursor matching escalate/replan/reassign."""
    placeholders = ",".join("?" for _ in REFLECTABLE_DECISION_TYPES)
    rows = conn.execute(
        f"""
        SELECT id, mission_id, iteration, decision_type, reasoning,
               subtask_id, target_agent_id, cost_usd, created_at
        FROM decisions
        WHERE id > ? AND decision_type IN ({placeholders})
        ORDER BY id ASC
        """,
        (after_id, *sorted(REFLECTABLE_DECISION_TYPES)),
    ).fetchall()
    return rows


@dataclass
class OutcomeGroup:
    """A cluster of failing outcomes sharing (task_type, agent_id)."""

    representative: sqlite3.Row  # lowest-scoring row in the group
    count: int
    member_ids: list[int]


@dataclass
class DecisionGroup:
    """A cluster of reflectable decisions sharing (mission_id, decision_type)."""

    representative: sqlite3.Row  # latest iteration in the group
    count: int
    member_ids: list[int]


def _group_outcomes(rows: list[sqlite3.Row]) -> list[OutcomeGroup]:
    """Group failing outcomes by (task_type, agent_id), keep the lowest-scoring.

    Prevents flooding IdeaForge when one broken mission produces N identical
    failure rows. Each group becomes a single idea that reports the cluster size.
    """
    groups: dict[tuple[str, str], OutcomeGroup] = {}
    for row in rows:
        key = (row["task_type"] or "general", row["agent_id"] or "unknown")
        score = float(row["composite_score"] or 0.0)
        existing = groups.get(key)
        if existing is None:
            groups[key] = OutcomeGroup(representative=row, count=1, member_ids=[int(row["id"])])
            continue
        existing.count += 1
        existing.member_ids.append(int(row["id"]))
        existing_score = float(existing.representative["composite_score"] or 0.0)
        if score < existing_score:
            existing.representative = row
    return list(groups.values())


def _group_decisions(rows: list[sqlite3.Row]) -> list[DecisionGroup]:
    """Group reflectable decisions by (mission_id, decision_type), keep latest iteration.

    Four replans on one mission become a single idea reporting the replan count.
    """
    groups: dict[tuple[str, str], DecisionGroup] = {}
    for row in rows:
        key = (row["mission_id"], row["decision_type"])
        existing = groups.get(key)
        if existing is None:
            groups[key] = DecisionGroup(representative=row, count=1, member_ids=[int(row["id"])])
            continue
        existing.count += 1
        existing.member_ids.append(int(row["id"]))
        if int(row["iteration"]) > int(existing.representative["iteration"]):
            existing.representative = row
    return list(groups.values())


def _outcome_group_to_idea(group: OutcomeGroup) -> dict[str, str | list[str]]:
    """Turn a failing-outcome cluster into a templated IdeaForge idea payload."""
    row = group.representative
    task_type = row["task_type"] or "general"
    agent_id = row["agent_id"] or "unknown"
    score = float(row["composite_score"] or 0.0)
    reasoning = _truncate(row["judge_reasoning"], 600)
    occurrences = (
        f" Pattern observed {group.count}x in this reflection window."
        if group.count > 1 else ""
    )

    title = (
        f"Capability gap: {task_type} task failed on agent '{agent_id}' "
        f"(score {score:.2f}"
        f"{f', x{group.count}' if group.count > 1 else ''})"
    )
    problem_statement = (
        f"Agent '{agent_id}' underperformed on a '{task_type}' subtask "
        f"(judge composite {score:.2f}, below the {COMPOSITE_FAILURE_THRESHOLD} threshold)."
        f"{occurrences} Worst-case judge reasoning: {reasoning}"
    )
    description = (
        f"Post-mortem signal from the agentic orchestrator. Mission "
        f"{row['mission_id']}, subtask {row['subtask_id']}, judged via "
        f"{row['judge_method']}. This idea is a candidate for building a "
        f"better tool, skill, or agent that would have succeeded where "
        f"'{agent_id}' failed.{occurrences}"
    )

    return {
        "title": title[:200],
        "description": description,
        "problem_statement": problem_statement,
        "target_audience": "ClaudeClaw orchestration",
        "tags": [
            "orchestrator-reflector",
            f"task-type:{task_type}",
            f"agent:{agent_id}",
            "failure-mode:low-composite",
            f"cluster-size:{group.count}",
        ],
        "source_signal_ids": [f"outcome:{mid}" for mid in group.member_ids],
    }


def _decision_group_to_idea(group: DecisionGroup) -> dict[str, str | list[str]]:
    """Turn a reflectable-decision cluster into a templated IdeaForge idea payload."""
    row = group.representative
    decision_type = row["decision_type"]
    reasoning = _truncate(row["reasoning"], 600)
    target_agent = row["target_agent_id"] or "none"
    occurrences = (
        f" Pattern observed {group.count}x on this mission."
        if group.count > 1 else ""
    )

    title = (
        f"Orchestrator decision ({decision_type}"
        f"{f' x{group.count}' if group.count > 1 else ''})"
        f": mission {row['mission_id']}"
    )
    problem_statement = (
        f"The reasoner chose to {decision_type} at iteration {row['iteration']} "
        f"(target agent: {target_agent}).{occurrences} Final reasoning: {reasoning}"
    )
    description = (
        f"Post-mortem signal from the agentic orchestrator. A '{decision_type}' "
        f"decision is evidence that the initial plan or agent assignment was "
        f"inadequate. Candidate for a tool/skill/agent idea that would have "
        f"avoided the {decision_type}.{occurrences}"
    )

    return {
        "title": title[:200],
        "description": description,
        "problem_statement": problem_statement,
        "target_audience": "ClaudeClaw orchestration",
        "tags": [
            "orchestrator-reflector",
            f"decision-type:{decision_type}",
            f"target-agent:{target_agent}",
            "failure-mode:reasoner-intervention",
            f"cluster-size:{group.count}",
        ],
        "source_signal_ids": [f"decision:{mid}" for mid in group.member_ids],
    }


def _reflect(
    conn: sqlite3.Connection,
    cursor: ReflectorCursor,
    ideaforge_db: Path | None,
    dry_run: bool,
) -> tuple[int, ReflectorCursor]:
    outcomes = _fetch_failing_outcomes(conn, cursor.last_outcome_id, COMPOSITE_FAILURE_THRESHOLD)
    decisions = _fetch_reflectable_decisions(conn, cursor.last_decision_id)

    outcome_groups = _group_outcomes(outcomes)
    decision_groups = _group_decisions(decisions)

    logger.info(
        "Reflector candidates: %d failing outcomes -> %d groups, "
        "%d decisions -> %d groups",
        len(outcomes), len(outcome_groups),
        len(decisions), len(decision_groups),
    )

    # Cursor must advance past ALL inspected rows, including within-cluster
    # duplicates we chose not to emit as separate ideas.
    max_outcome_id = max(
        (int(r["id"]) for r in outcomes),
        default=cursor.last_outcome_id,
    )
    max_decision_id = max(
        (int(r["id"]) for r in decisions),
        default=cursor.last_decision_id,
    )
    new_cursor = ReflectorCursor(
        last_outcome_id=max(cursor.last_outcome_id, max_outcome_id),
        last_decision_id=max(cursor.last_decision_id, max_decision_id),
    )

    if dry_run:
        for group in outcome_groups:
            row = group.representative
            logger.info(
                "[DRY RUN] outcome group %s/%s worst=%.2f x%d",
                row["task_type"], row["agent_id"],
                row["composite_score"], group.count,
            )
        for group in decision_groups:
            row = group.representative
            logger.info(
                "[DRY RUN] decision group %s/%s latest_iter=%d x%d",
                row["mission_id"], row["decision_type"],
                row["iteration"], group.count,
            )
        return (len(outcome_groups) + len(decision_groups), cursor)

    written = 0

    for group in outcome_groups:
        idea = _outcome_group_to_idea(group)
        gap_id = write_capability_gap(
            title=str(idea["title"]),
            description=str(idea["description"]),
            source_signal_ids=list(idea["source_signal_ids"]),  # type: ignore[arg-type]
            problem_statement=str(idea["problem_statement"]),
            target_audience=str(idea["target_audience"]),
            signal_source="orchestrator_reflector",
            db_path=ideaforge_db,
        )
        logger.info(
            "Wrote capability gap #%d from outcome group (%d members)",
            gap_id, group.count,
        )
        written += 1

    for group in decision_groups:
        idea = _decision_group_to_idea(group)
        gap_id = write_capability_gap(
            title=str(idea["title"]),
            description=str(idea["description"]),
            source_signal_ids=list(idea["source_signal_ids"]),  # type: ignore[arg-type]
            problem_statement=str(idea["problem_statement"]),
            target_audience=str(idea["target_audience"]),
            signal_source="orchestrator_reflector",
            db_path=ideaforge_db,
        )
        logger.info(
            "Wrote capability gap #%d from decision group (%d members)",
            gap_id, group.count,
        )
        written += 1

    return written, new_cursor


def run_agent(
    dry_run: bool = False,
    orchestrator_db: Path | None = None,
    ideaforge_db: Path | None = None,
    cursor_path: Path | None = None,
) -> str:
    """Run the orchestrator reflector.

    1. Read orchestrator.db for new failing outcomes and reflectable decisions.
    2. Template them into IdeaForge ideas (status='unscored').
    3. Advance the cursor so next run skips processed rows.

    Optional overrides are for tests.
    """
    db_path = orchestrator_db or ORCHESTRATOR_DB
    cur_path = cursor_path or CURSOR_PATH

    if not db_path.exists():
        msg = f"Orchestrator DB not found at {db_path} -- nothing to reflect"
        logger.warning(msg)
        return msg

    cursor = ReflectorCursor.load(cur_path)
    logger.info("Loaded cursor: %s", cursor)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        count, new_cursor = _reflect(conn, cursor, ideaforge_db, dry_run)
    finally:
        conn.close()

    if dry_run:
        return f"[DRY RUN] {count} reflector candidates"

    if count > 0:
        new_cursor.save(cur_path)
        logger.info("Advanced cursor: %s", new_cursor)

    return f"Wrote {count} orchestrator-reflector ideas to IdeaForge"
