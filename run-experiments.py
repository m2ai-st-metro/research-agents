#!/usr/bin/env python3
"""AutoResearch entry point.

Usage:
    python run-experiments.py                    # Full experiment run
    python run-experiments.py --dry-run          # Preview without running
    python run-experiments.py --agents arxiv     # Single agent
    python run-experiments.py --status           # Show ledger summary
    python run-experiments.py --validate         # Validate winners with Claude API
    python run-experiments.py --commit           # Commit validated winners
    python run-experiments.py --rollback-check   # Weekly rollback check
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure auto_research is importable
sys.path.insert(0, os.path.dirname(__file__))

# Load shared env for API keys
from pathlib import Path
env_shared = Path.home() / ".env.shared"
if env_shared.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(env_shared))
    except ImportError:
        # Manual env loading if python-dotenv not installed
        with open(env_shared) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch — Karpathy-style experiment loop for research-agents",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agents", type=str, default=None, help="Comma-separated agents")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show experiment summary")
    parser.add_argument("--validate", action="store_true", help="Validate winners with Claude")
    parser.add_argument("--commit", action="store_true", help="Commit validated winners")
    parser.add_argument("--rollback-check", action="store_true", help="Weekly rollback check")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.status:
        from auto_research.ledger import get_experiment_summary, init_db
        conn = init_db()
        summary = get_experiment_summary(conn)
        conn.close()
        print("AutoResearch Experiment Ledger")
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Winners (>15%):    {summary['winners']}")
        print(f"  Committed:         {summary['committed']}")
        print(f"  Rolled back:       {summary['rolled_back']}")
        print(f"  Claude validated:  {summary['claude_validated']}")
        return

    if args.validate:
        from auto_research.claude_validator import validate_top_winners
        from auto_research.ledger import init_db
        conn = init_db()
        validated = validate_top_winners(conn)
        conn.close()
        print(f"Validated {len(validated)} winners with Claude API")
        return

    if args.commit:
        from auto_research.committer import commit_winner
        from auto_research.ledger import get_winners, init_db
        conn = init_db()
        winners = get_winners(conn)
        validated_winners = [w for w in winners if w["claude_validated"]]

        if not validated_winners:
            print("No validated winners to commit")
            conn.close()
            return

        committed = 0
        for w in validated_winners:
            success = commit_winner(
                conn=conn,
                experiment_id=w["id"],
                agent=w["agent"],
                param_name=w["param_name"],
                old_query=w["baseline_value"],
                new_query=w["variant_value"],
                improvement_pct=w["improvement_pct"],
                dry_run=args.dry_run,
            )
            if success:
                committed += 1

        conn.close()
        print(f"Committed {committed}/{len(validated_winners)} winning mutations")
        return

    if args.rollback_check:
        from auto_research.committer import check_weekly_rollback
        from auto_research.ledger import init_db
        conn = init_db()
        # TODO: compute current production NDR from IdeaForge DB
        print("Rollback check requires production NDR — not yet integrated")
        conn.close()
        return

    # Default: run experiments
    from auto_research.runner import run_experiments
    agents = args.agents.split(",") if args.agents else None
    run_experiments(agents=agents, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
