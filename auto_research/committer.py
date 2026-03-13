"""Auto-committer — commits winning mutations to config.py and handles rollbacks.

Modifies research-agents/src/research_agents/config.py in-place,
then commits and pushes via git.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sqlite3
from pathlib import Path

from .config import AGENT_QUERY_KEYS, ROLLBACK_THRESHOLD
from .ledger import (
    get_committed_this_week,
    get_last_weekly_baseline,
    mark_committed,
    mark_rolled_back,
    save_weekly_baseline,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "research_agents"
    / "config.py"
)


def _read_config() -> str:
    """Read current config.py contents."""
    return CONFIG_PATH.read_text()


def _write_config(content: str) -> None:
    """Write modified config.py."""
    CONFIG_PATH.write_text(content)


def _replace_query_in_config(
    config_text: str,
    query_key: str,
    old_query: str,
    new_query: str,
) -> str | None:
    """Replace a specific query string in the config list.

    Returns modified config text, or None if replacement failed.
    """
    # Escape for regex
    old_escaped = re.escape(old_query)

    # Look for the query inside the named list
    pattern = re.compile(
        rf'((?:{re.escape(query_key)}[^]]*?))'
        rf'("|\'){old_escaped}\2',
        re.DOTALL,
    )

    # Simpler approach: direct string replacement within quotes
    for quote in ['"', "'"]:
        target = f'{quote}{old_query}{quote}'
        replacement = f'{quote}{new_query}{quote}'
        if target in config_text:
            return config_text.replace(target, replacement, 1)

    logger.warning(
        "Could not find query '%s' in config key '%s'",
        old_query,
        query_key,
    )
    return None


def _git_commit_and_push(message: str) -> str | None:
    """Git add, commit, and push config.py. Returns commit SHA or None."""
    try:
        repo_root = CONFIG_PATH.parent.parent.parent.parent
        subprocess.run(
            ["git", "add", str(CONFIG_PATH)],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()

        subprocess.run(
            ["git", "push"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        logger.info("Committed and pushed: %s (%s)", message, sha[:8])
        return sha

    except subprocess.CalledProcessError as e:
        logger.error("Git operation failed: %s", e.stderr)
        return None


def commit_winner(
    conn: sqlite3.Connection,
    experiment_id: int,
    agent: str,
    param_name: str,
    old_query: str,
    new_query: str,
    improvement_pct: float,
    dry_run: bool = False,
) -> bool:
    """Commit a winning query mutation to config.py.

    Args:
        conn: Ledger DB connection.
        experiment_id: Experiment ID in ledger.
        agent: Agent name.
        param_name: Parameter name (e.g., ARXIV_SEARCH_QUERIES[2]).
        old_query: The query being replaced.
        new_query: The winning variant query.
        improvement_pct: How much better the variant was.
        dry_run: If True, show what would change without modifying files.

    Returns:
        True if committed successfully.
    """
    query_key = AGENT_QUERY_KEYS.get(agent)
    if not query_key:
        logger.error("No query key for agent '%s'", agent)
        return False

    config_text = _read_config()
    modified = _replace_query_in_config(config_text, query_key, old_query, new_query)

    if modified is None:
        logger.error("Failed to replace query in config.py")
        return False

    if dry_run:
        logger.info(
            "[DRY RUN] Would replace in %s:\n  OLD: '%s'\n  NEW: '%s'",
            query_key,
            old_query,
            new_query,
        )
        return True

    _write_config(modified)

    commit_msg = (
        f"auto-research: update {query_key} query (+{improvement_pct:.0%} NDR)\n\n"
        f"Agent: {agent}\n"
        f"Old query: {old_query}\n"
        f"New query: {new_query}\n"
        f"Improvement: {improvement_pct:.1%} non-dismiss rate\n"
        f"Experiment ID: {experiment_id}\n\n"
        f"Co-Authored-By: AutoResearch <noreply@autoresearch.local>"
    )

    sha = _git_commit_and_push(commit_msg)
    if sha:
        mark_committed(conn, experiment_id, sha)
        return True

    # Revert file change if git failed
    _write_config(config_text)
    return False


def check_weekly_rollback(
    conn: sqlite3.Connection,
    current_ndr: float,
    current_avg_score: float,
    dry_run: bool = False,
) -> bool:
    """Check if this week's metrics warrant rolling back committed changes.

    Compares current production metrics against last week's baseline.
    If NDR dropped > ROLLBACK_THRESHOLD, reverts all commits from this week.

    Returns True if rollback was triggered.
    """
    last_baseline = get_last_weekly_baseline(conn)
    if last_baseline is None:
        logger.info("No previous baseline — saving current as baseline")
        save_weekly_baseline(conn, current_ndr, current_avg_score, {})
        return False

    baseline_ndr = last_baseline["overall_ndr"]
    if baseline_ndr == 0:
        save_weekly_baseline(conn, current_ndr, current_avg_score, {})
        return False

    ndr_change = (current_ndr - baseline_ndr) / baseline_ndr

    if ndr_change < -ROLLBACK_THRESHOLD:
        logger.warning(
            "Weekly NDR dropped %.1f%% (%.1f%% -> %.1f%%). Triggering rollback.",
            ndr_change * 100,
            baseline_ndr * 100,
            current_ndr * 100,
        )

        committed = get_committed_this_week(conn)
        if not committed:
            logger.info("No committed changes this week to roll back")
            save_weekly_baseline(conn, current_ndr, current_avg_score, {})
            return False

        if dry_run:
            logger.info(
                "[DRY RUN] Would roll back %d commits from this week",
                len(committed),
            )
            return True

        # Revert each committed change
        config_text = _read_config()
        reverted_ids: list[int] = []
        for row in reversed(committed):  # Reverse chronological order
            # Swap new_query back to old_query
            result = _replace_query_in_config(
                config_text,
                AGENT_QUERY_KEYS.get(row["agent"], ""),
                row["variant_value"],  # new becomes old
                row["baseline_value"],  # old becomes new
            )
            if result:
                config_text = result
                reverted_ids.append(row["id"])
                logger.info(
                    "Reverted [%s]: '%s' -> '%s'",
                    row["agent"],
                    row["variant_value"],
                    row["baseline_value"],
                )

        if reverted_ids:
            _write_config(config_text)
            sha = _git_commit_and_push(
                f"auto-research: weekly rollback ({len(reverted_ids)} changes reverted)\n\n"
                f"NDR dropped {ndr_change:.1%} below threshold ({ROLLBACK_THRESHOLD:.0%})"
            )
            if sha:
                mark_rolled_back(conn, reverted_ids)

        return True

    logger.info(
        "Weekly NDR change: %+.1f%% (%.1f%% -> %.1f%%). No rollback needed.",
        ndr_change * 100,
        baseline_ndr * 100,
        current_ndr * 100,
    )
    save_weekly_baseline(conn, current_ndr, current_avg_score, {})
    return False
