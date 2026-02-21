# BLUEPRINT.md - Research Agents

## Phase Tracking

### Phase 1: ResearchSignal Contract + ContractStore Extension
- [x] Create `ResearchSignal` Pydantic model in snow-town
- [x] Add `SignalSource`, `SignalRelevance` enums
- [x] Extend ContractStore with `research_signals` table
- [x] Add write/read/query/update methods
- [x] Tests pass (71/71)

### Phase 2: Project Scaffold
- [x] Directory structure created
- [x] pyproject.toml + hatchling build
- [x] config.py with queries, cadences, model settings
- [x] claude_client.py for relevance assessment
- [x] signal_writer.py with ContractStore integration
- [x] runner.py CLI with Typer
- [x] run-agents.sh with lock file + env sourcing
- [x] Tests for signal_writer (4/4)
- [x] `pip install -e ".[dev]"` verified
- [x] `research-agents status` verified

### Phase 3: ArXiv/Paper Scanner Agent
- [x] arxiv_scanner.py with arXiv API integration
- [x] Deduplication against existing signals
- [x] Claude relevance assessment
- [x] Dry-run support
- [x] Tests (7/7)

### Phase 4: Tool/Library Monitor Agent
- [x] tool_monitor.py with GitHub search integration
- [x] Persona-aware tagging
- [x] Tests (6/6)

### Phase 5: Adjacent Domain Watcher Agent
- [x] domain_watcher.py with HN Algolia search
- [x] HIGH-only relevance filter
- [x] Every-3-day cadence config
- [x] Tests (5/5)

### Phase 6: Sky-Lynx Integration
- [x] research_reader.py (mirrors ideaforge_reader.py)
- [x] Integrate into analyzer.py
- [x] Integrate into claude_client.py prompt
- [x] Update Sky-Lynx CLAUDE.md
- [x] Tests (11/11 new, 49/49 total)

### Phase 7: Idea Surfacer Agent
- [x] idea_surfacer.py
- [x] Write to caught_ideas.db with machine-surfaced tag
- [x] Mark consumed signals
- [x] Tests (6/6)

### Phase 8: Dashboard Integration
- [x] loop_status.py research signal metrics
- [x] API router for research signals (GET /signals, GET /summary)
- [x] Ecosystem graph update (4th node + edge)
- [x] All Snow-Town tests pass (71/71)

### Phase 9: Cron Orchestration
- [x] Cron config files (cron/research-agents)
- [x] Log rotation (cron/logrotate-research-agents)
- [x] Log directory created (/var/log/research-agents/)

## Weekly Data Flow

```
Mon-Sat 5AM:  arxiv + tool-monitor → ResearchSignals
Wed/Sat 5AM:  domain-watcher → ResearchSignals
Sat 11PM:     idea-surfacer → caught_ideas.db
Sun 2AM:      Snow-Town loop (SL now reads research signals too)
```

## Test Summary

| Project | Tests | Status |
|---------|-------|--------|
| Snow-Town | 71 | All passing |
| Sky-Lynx | 49 | All passing |
| Research Agents | 28 | All passing |
