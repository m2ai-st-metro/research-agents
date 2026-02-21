# Research Agents

Ambient research intelligence for the Snow-Town feedback loop. Watches industry patterns, monitors tools/libraries, and tracks adjacent domains so the factory self-improves from richer context.

## Architecture

Research agents are **signal producers** that write `ResearchSignal` contracts to Snow-Town's ContractStore. Sky-Lynx reads these signals during weekly analysis, and the idea-surfacer synthesizes them into actionable project ideas.

```
ArXiv Scanner ─────┐
Tool Monitor ──────┼──→ ResearchSignal (Snow-Town) ──→ Sky-Lynx analysis
Domain Watcher ────┘                                    │
                                                        ▼
Idea Surfacer ←── synthesizes signals ──→ caught_ideas.db (Ultra Magnus)
```

## Setup

```bash
cd ~/projects/research-agents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires `~/.env.shared` with `ANTHROPIC_API_KEY`.

## Usage

```bash
# Run a specific agent
research-agents run arxiv
research-agents run tool-monitor
research-agents run domain-watch
research-agents run idea-surfacer

# Dry run (no API calls or writes)
research-agents run arxiv --dry-run

# Show configuration and signal counts
research-agents status

# Shell wrapper (for cron)
./run-agents.sh arxiv tool-monitor
./run-agents.sh --dry-run arxiv
```

## Agents

| Agent | Source | Cadence | Min Relevance |
|-------|--------|---------|---------------|
| `arxiv` | arXiv/Semantic Scholar papers | Daily 5 AM | medium |
| `tool-monitor` | MCP servers, frameworks, AI tools | Daily 5 AM | medium |
| `domain-watch` | Healthcare AI, solo dev, workflows | Every 3 days | high |
| `idea-surfacer` | Synthesis of all signals | Saturday 11 PM | n/a |

## Testing

```bash
pytest tests/ -v
```
