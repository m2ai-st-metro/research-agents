# CLAUDE.md - Research Agents

## Quick Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
research-agents status
```

## Project Purpose

Ambient research intelligence for ST Factory's self-correcting feedback loop. Four agents scan papers, tools, adjacent domains, and synthesize findings into ideas — all writing to ST Factory's ContractStore as `ResearchSignal` records.

## Architecture

- Signal producers (agents) write `ResearchSignal` to ST Factory ContractStore
- Sky-Lynx reads signals via `research_reader.py` during weekly analysis
- Idea surfacer writes directly to ultra-magnus `caught_ideas.db`
- Claude API (Sonnet) handles relevance assessment

## Data Flow

```
Agents → ResearchSignal (JSONL + SQLite in st-factory/data/)
Sky-Lynx reads signals → includes in weekly analysis prompt
Idea surfacer → caught_ideas.db (ultra-magnus)
```

## Key Patterns

- `signal_writer.py` imports ContractStore via sys.path (same as Sky-Lynx)
- `config.py` holds all queries, thresholds, model settings
- `run-agents.sh` mirrors IdeaForge's lock file + env sourcing pattern
- Each agent has `run_agent(dry_run: bool) -> str` entry point

## Dependencies

- ST Factory contracts (via sys.path, not pip)
- Ultra-Magnus caught_ideas.db (direct SQLite for idea surfacer)
- Anthropic API for relevance assessment

## Cron Schedule

```
Daily 5 AM:      arxiv + tool-monitor
Every 3 days:    domain-watch
Saturday 11 PM:  idea-surfacer
```
