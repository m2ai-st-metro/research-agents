# CLAUDE.md - Research Agents

## Quick Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
research-agents status
research-agents run rss --dry-run  # Test a single agent
```

## Project Purpose

Skill-foundry intelligence pipeline for ST Metro. 9 agents scan for MCP ecosystem gaps, workflow patterns, agent infrastructure, and unwrapped API surfaces -- writing to ST Records' ContractStore as `ResearchSignal` records. Idea surfacer synthesizes signals into buildable skill/MCP/workflow ideas.

**Mission (2026-04-05):** Identify what MCP servers, agent skills, workflow tools, and pipeline components should exist but don't yet. Quality over volume.

## Architecture

- Signal producers (agents) write `ResearchSignal` to ST Records ContractStore
- Sky-Lynx reads signals via `research_reader.py` during weekly analysis
- Idea surfacer writes to IdeaForge `ideaforge.db` (status='unscored')
- IdeaForge scores + classifies (skill-fit weighted) -> Metroplex triages

## Agent Registry (9 active + 2 utility)

| Agent | LLM | Focus | Cadence |
|-------|-----|-------|---------|
| tool-monitor | Ollama (AlienPC) | MCP repos, agent SDKs on GitHub | Daily 5 AM |
| rss | Ollama (AlienPC) | Dev.to MCP/Agents/Automation, Simon Willison, Latent Space | Daily 5 AM |
| perplexity | Sonar Pro (OpenRouter) | MCP skill gaps, missing integrations | Daily 6 AM |
| chatgpt | GPT-5.4 (OpenAI) | Workflow patterns lacking tooling | Every 3 days |
| gemini-research | Gemini 3 Flash (Google) | New MCP/agent releases <7 days | Daily 6:15 AM |
| youtube | Gemini + Ollama | Agent/MCP tutorial content | Daily 5:15 AM |
| reddit | Ollama (AlienPC) | r/ClaudeAI, r/LocalLLaMA, r/devtools pain points | Daily 6:45 AM |
| trend-analyzer | Ollama (AlienPC) | Weekly synthesis with Skill Gap Radar + Build Queue | Weekly |
| idea-surfacer | Nemotron-3 (DeepInfra) | Synthesize signals into 2-4 week MVP ideas | Every 3 days |
| manual-signal | Ollama | Matthew's manual URL/topic ingestion | Manual |
| ideaforge-writer | N/A | Helper: writes ideas to IdeaForge DB | Internal |

### Retired Agents (2026-04-05)
arxiv_scanner, domain_watcher, github_trending, producthunt_scanner -- not aligned with skill-foundry mission. Tests in `tests/retired/`.

## Key Patterns

- `signal_writer.py` imports ContractStore via sys.path (same as Sky-Lynx)
- `config.py` holds all queries, thresholds, model settings
- `run-agents.sh` mirrors IdeaForge's lock file + env sourcing pattern
- Each agent has `run_agent(dry_run: bool) -> str` entry point
- Anti-monoculture: different LLM per agent to avoid echo chambers

## Infrastructure

- **Ollama default: AlienPC GPU** (`http://10.0.0.35:11434`, qwen2.5:14b, RTX 5080)
- ProBook CPU is fallback only (~124s/assessment vs ~7s on GPU)
- If AlienPC is off, override: `OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=qwen2.5:7b-instruct`
- Firecrawl enrichment enabled for tool-monitor README scraping

## Dependencies

- ST Records contracts (via sys.path, not pip)
- IdeaForge ideaforge.db (direct SQLite for idea surfacer, config: IDEAFORGE_DB)
- Ollama on AlienPC for relevance assessment (5 agents)
- Paid APIs: OpenRouter (Perplexity), OpenAI (ChatGPT), Google (Gemini), DeepInfra (Nemotron-3)

## Cron Schedule

```
Daily 5:00 AM:     tool-monitor + rss
Daily 5:15 AM:     youtube
Daily 6:00 AM:     perplexity
Daily 6:15 AM:     gemini-research
Daily 6:45 AM:     reddit
Every 3 days:      chatgpt (6:30 AM), idea-surfacer (11 PM)
Weekly (Sat):      trend-analyzer (10 PM)
```

Install: `sudo cp cron/research-agents /etc/cron.d/research-agents`
