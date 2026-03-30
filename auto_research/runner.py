"""Experiment runner — main orchestrator for AutoResearch experiments.

Entry point: python -m auto_research.runner [--dry-run] [--agents arxiv,tool_monitor]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .committer import commit_winner
from .config import (
    AGENT_QUERY_KEYS,
    AUTO_COMMIT_ENABLED,
    EXPERIMENT_AGENTS,
    IMPROVEMENT_THRESHOLD,
    MAX_CLAUDE_VALIDATIONS,
)
from .evaluator import Comparison, compare
from .ledger import (
    get_winners,
    init_db,
    log_experiment,
)
from .mini_pipeline import ExperimentResult, run_experiment
from .mutator import generate_variant, select_query_to_mutate
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def _load_agent_queries(agent: str) -> list[str]:
    """Load current queries for an agent from research-agents config."""
    # Import dynamically to get current state
    config_path = Path(__file__).resolve().parent.parent / "src" / "research_agents"
    if str(config_path.parent) not in sys.path:
        sys.path.insert(0, str(config_path.parent))

    # Re-import to pick up any changes
    import importlib
    try:
        import research_agents.config as ra_config
        importlib.reload(ra_config)
    except ImportError:
        logger.error("Cannot import research_agents.config — is src/ accessible?")
        return []

    key = AGENT_QUERY_KEYS.get(agent)
    if not key:
        logger.error("No query key mapping for agent '%s'", agent)
        return []

    queries = getattr(ra_config, key, [])

    # RSS is a list of dicts, extract names for display
    if agent == "rss" and queries and isinstance(queries[0], dict):
        return [f["name"] for f in queries]

    return list(queries)


def run_experiments(
    agents: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    rounds: int = 1,
) -> list[Comparison]:
    """Run multiple rounds of AutoResearch experiments.

    Each round tests one random query per agent with a new variant.

    Args:
        agents: List of agents to experiment on (default: EXPERIMENT_AGENTS).
        dry_run: If True, show what would happen without running.
        verbose: If True, log detailed experiment info.
        rounds: Number of rounds to run (each round tests all agents).

    Returns:
        List of Comparison results across all rounds.
    """
    agents = agents or EXPERIMENT_AGENTS
    comparisons: list[Comparison] = []

    # Check Ollama availability
    client = OllamaClient()
    if not dry_run and not client.is_available():
        logger.error(
            "Ollama is not available at %s or model '%s' not loaded. "
            "Start Ollama and pull the model first.",
            client.base_url,
            client.model,
        )
        return comparisons

    # Initialize ledger
    conn = init_db()

    try:
        for round_num in range(1, rounds + 1):
            if rounds > 1:
                logger.info("")
                logger.info("###  ROUND %d / %d  ###", round_num, rounds)
                logger.info("")

            for agent in agents:
                logger.info("=" * 60)
                logger.info("Experimenting on agent: %s", agent)
                logger.info("=" * 60)

                queries = _load_agent_queries(agent)
                if not queries:
                    logger.warning("No queries found for agent '%s', skipping", agent)
                    continue

                # Select query to mutate
                query_idx, baseline_query = select_query_to_mutate(queries, agent)
                param_name = f"{AGENT_QUERY_KEYS.get(agent, agent)}[{query_idx}]"

                logger.info("Baseline query: '%s'", baseline_query)

                if dry_run:
                    logger.info("[DRY RUN] Would generate variant and run experiment")
                    continue

                # Generate variant
                variant_query = generate_variant(
                    current_query=baseline_query,
                    agent=agent,
                    client=client,
                    all_queries=queries,
                )
                logger.info("Variant query:  '%s'", variant_query)

                # Run baseline
                logger.info("Running baseline experiment...")
                t0 = time.time()
                baseline_result = run_experiment(
                    query=baseline_query,
                    agent=agent,
                    client=client,
                )
                baseline_time = time.time() - t0
                logger.info(
                    "Baseline: %d signals found, %d relevant, %d ideas, NDR=%.1f%% (%.1fs)",
                    baseline_result.signals_found,
                    baseline_result.signals_relevant,
                    baseline_result.ideas_total,
                    baseline_result.non_dismiss_rate * 100,
                    baseline_time,
                )

                # Rate limit between API calls
                time.sleep(3)

                # Run variant
                logger.info("Running variant experiment...")
                t0 = time.time()
                variant_result = run_experiment(
                    query=variant_query,
                    agent=agent,
                    client=client,
                )
                variant_time = time.time() - t0
                logger.info(
                    "Variant:  %d signals found, %d relevant, %d ideas, NDR=%.1f%% (%.1fs)",
                    variant_result.signals_found,
                    variant_result.signals_relevant,
                    variant_result.ideas_total,
                    variant_result.non_dismiss_rate * 100,
                    variant_time,
                )

                # Compare
                comparison = compare(
                    agent=agent,
                    param_name=param_name,
                    baseline=baseline_result,
                    variant=variant_result,
                )
                comparisons.append(comparison)

                logger.info("Result: %s", comparison.reason)

                # Log to ledger
                exp_id = log_experiment(
                    conn=conn,
                    agent=agent,
                    param_name=param_name,
                    baseline_value=baseline_query,
                    variant_value=variant_query,
                    baseline_signals=baseline_result.signals_relevant,
                    variant_signals=variant_result.signals_relevant,
                    baseline_ndr=baseline_result.non_dismiss_rate,
                    variant_ndr=variant_result.non_dismiss_rate,
                    baseline_avg_score=baseline_result.avg_weighted_score,
                    variant_avg_score=variant_result.avg_weighted_score,
                    improvement_pct=comparison.improvement_pct,
                    status="completed" if comparison.is_valid else "insufficient_data",
                    notes=comparison.reason,
                )

                # Commit winners (gated by AUTO_COMMIT_ENABLED)
                if comparison.is_winner:
                    if AUTO_COMMIT_ENABLED:
                        logger.info(
                            "AUTO-COMMIT: %s query '%s' -> '%s' (+%.1f%% NDR)",
                            agent, baseline_query, variant_query,
                            comparison.improvement_pct * 100,
                        )
                        committed = commit_winner(
                            conn=conn,
                            experiment_id=exp_id,
                            agent=agent,
                            param_name=param_name,
                            old_query=baseline_query,
                            new_query=variant_query,
                            improvement_pct=comparison.improvement_pct,
                            dry_run=dry_run,
                        )
                        if committed:
                            logger.info("Committed successfully.")
                        else:
                            logger.warning("Commit failed — query may have changed since experiment started.")
                    else:
                        logger.info(
                            "WINNER (pending review): %s query '%s' -> '%s' (+%.1f%% NDR). "
                            "Set AUTO_COMMIT_ENABLED=True in config to auto-apply.",
                            agent, baseline_query, variant_query,
                            comparison.improvement_pct * 100,
                        )

                # Rate limit between agents
                time.sleep(5)

    finally:
        conn.close()

    # Summary
    winners = [c for c in comparisons if c.is_winner]
    logger.info("=" * 60)
    logger.info(
        "Experiment run complete: %d rounds, %d experiments, %d winners",
        rounds,
        len(comparisons),
        len(winners),
    )
    for w in winners:
        logger.info(
            "  WINNER [%s]: '%s' -> '%s' (NDR +%.1f%%)",
            w.agent,
            w.baseline_query,
            w.variant_query,
            w.improvement_pct * 100,
        )

    return comparisons


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoResearch — Karpathy-style experiment loop for research-agents",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without running experiments",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agents to experiment on",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show experiment ledger summary and exit",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of experiment rounds to run (default: 1)",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.status:
        from .ledger import get_experiment_summary, init_db
        conn = init_db()
        summary = get_experiment_summary(conn)
        conn.close()
        print(f"AutoResearch Experiment Ledger")
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Winners (>15%):    {summary['winners']}")
        print(f"  Committed:         {summary['committed']}")
        print(f"  Rolled back:       {summary['rolled_back']}")
        print(f"  Claude validated:  {summary['claude_validated']}")
        return

    agents = args.agents.split(",") if args.agents else None
    run_experiments(
        agents=agents,
        dry_run=args.dry_run,
        verbose=args.verbose,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
