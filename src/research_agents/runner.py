"""CLI runner for research agents."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import CADENCE, SNOW_TOWN_ROOT, ULTRA_MAGNUS_DB

app = typer.Typer(name="research-agents", help="Ambient research intelligence for Snow-Town.")
console = Console()
logger = logging.getLogger(__name__)

# Agent registry — populated as agents are implemented
AGENTS: dict[str, str] = {
    "arxiv": "research_agents.agents.arxiv_scanner",
    "tool-monitor": "research_agents.agents.tool_monitor",
    "domain-watch": "research_agents.agents.domain_watcher",
    "idea-surfacer": "research_agents.agents.idea_surfacer",
}


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def run(
    agent: str = typer.Argument(..., help="Agent to run: arxiv, tool-monitor, domain-watch, idea-surfacer"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without API calls or writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run a specific research agent."""
    _setup_logging(verbose)

    if agent not in AGENTS:
        console.print(f"[red]Unknown agent: {agent}[/red]")
        console.print(f"Available: {', '.join(AGENTS.keys())}")
        raise typer.Exit(1)

    module_path = AGENTS[agent]
    try:
        module = __import__(module_path, fromlist=["run_agent"])
        run_fn = getattr(module, "run_agent")
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Agent '{agent}' not yet implemented: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Running agent: {agent}[/bold] (dry_run={dry_run})")
    result = run_fn(dry_run=dry_run)
    console.print(f"[green]Agent '{agent}' complete. {result}[/green]")


@app.command()
def run_all(
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run all agents sequentially."""
    _setup_logging(verbose)
    for agent_name in AGENTS:
        console.print(f"\n[bold]>>> {agent_name}[/bold]")
        try:
            module_path = AGENTS[agent_name]
            module = __import__(module_path, fromlist=["run_agent"])
            run_fn = getattr(module, "run_agent")
            result = run_fn(dry_run=dry_run)
            console.print(f"[green]  {result}[/green]")
        except (ImportError, AttributeError):
            console.print(f"[yellow]  Skipping (not yet implemented)[/yellow]")
        except Exception as e:
            console.print(f"[red]  Error: {e}[/red]")


@app.command()
def status() -> None:
    """Show configuration and signal summary."""
    _setup_logging()

    table = Table(title="Research Agents Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Snow-Town root", str(SNOW_TOWN_ROOT))
    table.add_row("Snow-Town DB", str(SNOW_TOWN_ROOT / "data" / "persona_metrics.db"))
    table.add_row("Ultra-Magnus DB", str(ULTRA_MAGNUS_DB))

    for agent_name, cadence in CADENCE.items():
        table.add_row(f"  {agent_name} cadence", cadence)

    console.print(table)

    # Try to load signal counts
    try:
        sys.path.insert(0, str(SNOW_TOWN_ROOT))
        from contracts import ContractStore
        store = ContractStore()
        try:
            all_signals = store.query_signals(limit=10000)
            console.print(f"\n[bold]Signals in store:[/bold] {len(all_signals)}")

            by_source: dict[str, int] = {}
            by_relevance: dict[str, int] = {}
            for s in all_signals:
                by_source[s.source.value] = by_source.get(s.source.value, 0) + 1
                by_relevance[s.relevance.value] = by_relevance.get(s.relevance.value, 0) + 1

            if by_source:
                for src, cnt in sorted(by_source.items()):
                    console.print(f"  {src}: {cnt}")
            if by_relevance:
                console.print("[bold]By relevance:[/bold]")
                for rel, cnt in sorted(by_relevance.items()):
                    console.print(f"  {rel}: {cnt}")
        finally:
            store.close()
    except Exception as e:
        console.print(f"[yellow]Could not load signals: {e}[/yellow]")


if __name__ == "__main__":
    app()
