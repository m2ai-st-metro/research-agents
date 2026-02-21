# AGENTS.md - Research Agents

## Build & Run

```bash
# Setup
cd ~/projects/research-agents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run agents
research-agents run arxiv --dry-run
research-agents run tool-monitor
research-agents run domain-watch
research-agents run idea-surfacer
research-agents status

# Tests
pytest tests/ -v

# Shell wrapper (cron-compatible)
./run-agents.sh arxiv tool-monitor
```

## Environment

Requires `~/.env.shared` with:
- `ANTHROPIC_API_KEY` - for Claude relevance assessment

Optional overrides:
- `SNOW_TOWN_ROOT` - defaults to `~/projects/st-factory`
- `IDEA_CATCHER_DB` - defaults to `~/projects/ultra-magnus/idea-catcher/data/caught_ideas.db`

## Agent Entry Points

Each agent module exposes `run_agent(dry_run: bool = False) -> str`:
- `src/research_agents/agents/arxiv_scanner.py`
- `src/research_agents/agents/tool_monitor.py`
- `src/research_agents/agents/domain_watcher.py`
- `src/research_agents/agents/idea_surfacer.py`
