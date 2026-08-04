---
agent: ravage (claude-sonnet-4-6)
date: 2026-08-04
runner_invocation: python -m auto_research.runner --rounds 2
runtime_window: 2026-08-04 02:01:30-02:06:56 CDT. Full run via PYTHONPATH subshell (note: .venv, not venv). 5/6 experiments persisted; round 2 gemini_research errored on baseline parse.
status: COMPLETE (5/6 experiments; 1 error) -- 1 winner, 1 commit.
---

# AutoResearch Nightly Batch -- 2026-08-04

## Hive-Mind Payload (machine-readable)

```json
{
  "sink": "logs/autoresearch-nightly-2026-08-04.md",
  "date": "2026-08-04",
  "run_timestamp": "2026-08-04T02:01:30",
  "runner_invocation": "python -m auto_research.runner --rounds 2",
  "rounds_requested": 2,
  "rounds_completed": "2 (5 of 6 expected experiments persisted: ledger ids 4685-4689; round 2 gemini_research errored before persisting)",
  "experiments_ran": 5,
  "experiments_expected_full_run": 6,
  "experiments_missing": 1,
  "experiment_results": [
    {
      "ledger_id": 4685,
      "timestamp": "2026-08-04T02:02:00.992226",
      "agent": "tool_monitor",
      "param_name": "TOOL_SEARCH_QUERIES[5]",
      "baseline_value": "MCP SDK typescript python client binding",
      "variant_value": "MCP client-server integration SDK TypeScript Python bindings",
      "baseline_signals": 2,
      "variant_signals": 0,
      "baseline_ndr": 0.667,
      "variant_ndr": 0.0,
      "improvement_pct": 0.0,
      "status": "insufficient_data",
      "notes": "Baseline has only 2 signals (min: 3)",
      "is_winner": false
    },
    {
      "ledger_id": 4686,
      "timestamp": "2026-08-04T02:03:27.728670",
      "agent": "youtube",
      "param_name": "YOUTUBE_SEARCH_QUERIES[3]",
      "baseline_value": "AI workflow automation with agent pipelines 2026",
      "variant_value": "AI driven workflow orchestration with autonomous agents 2026",
      "baseline_signals": 10,
      "variant_signals": 10,
      "baseline_ndr": 0.667,
      "variant_ndr": 1.0,
      "improvement_pct": 0.50,
      "status": "completed",
      "notes": "Winner: NDR improved 50.0% (threshold: 20%)",
      "is_winner": true
    },
    {
      "ledger_id": 4687,
      "timestamp": "2026-08-04T02:05:01.281031",
      "agent": "gemini_research",
      "param_name": "GEMINI_RESEARCH_QUERIES[2]",
      "baseline_value": "Search for GitHub repositories with recent star increases and active development on Model Context Protocol and AI agent skill enhancements in the last week",
      "variant_value": "Search for GitHub repositories showing recent activity and star growth in Model Context Protocol and AI agent skill development over the last week",
      "baseline_signals": 5,
      "variant_signals": 5,
      "baseline_ndr": 1.0,
      "variant_ndr": 1.0,
      "improvement_pct": 0.0,
      "status": "completed",
      "notes": "NDR improved 0.0% but avg score dropped (6.1 -> 6.0). Guardrail triggered.",
      "is_winner": false
    },
    {
      "ledger_id": 4688,
      "timestamp": "2026-08-04T02:05:10.509738",
      "agent": "tool_monitor",
      "param_name": "TOOL_SEARCH_QUERIES[1]",
      "baseline_value": "MCP bridge service API wrapper",
      "variant_value": "MCP service connector API integration examples",
      "baseline_signals": 0,
      "variant_signals": 0,
      "baseline_ndr": 0.0,
      "variant_ndr": 0.0,
      "improvement_pct": 0.0,
      "status": "insufficient_data",
      "notes": "Baseline has only 0 signals (min: 3)",
      "is_winner": false
    },
    {
      "ledger_id": 4689,
      "timestamp": "2026-08-04T02:06:38.221167",
      "agent": "youtube",
      "param_name": "YOUTUBE_SEARCH_QUERIES[1]",
      "baseline_value": "unveiling new capabilities of AI agent framework 2026",
      "variant_value": "exploring advanced features of AI agent framework launch 2026",
      "baseline_signals": 10,
      "variant_signals": 10,
      "baseline_ndr": 1.0,
      "variant_ndr": 1.0,
      "improvement_pct": 0.0,
      "status": "completed",
      "notes": "NDR improved 0.0% but avg score dropped (6.4 -> 5.8). Guardrail triggered.",
      "is_winner": false
    }
  ],
  "errored_experiments": [
    {
      "round": 2,
      "agent": "gemini_research",
      "error_class": "AttributeError",
      "error_msg": "'list' object has no attribute 'get'",
      "location": "auto_research/mini_pipeline.py:324 _parse_paid_signals",
      "note": "Gemini baseline ran (got response) but the result was a list instead of a dict; _parse_paid_signals expects dict with 'signals' key. No ledger row written -- experiment never persisted.",
      "is_winner": false
    }
  ],
  "winners_count": 1,
  "winners": [
    {
      "agent": "youtube",
      "param_name": "YOUTUBE_SEARCH_QUERIES[3]",
      "old_query": "AI workflow automation with agent pipelines 2026",
      "new_query": "AI driven workflow orchestration with autonomous agents 2026",
      "improvement_pct": 0.50,
      "ledger_id": 4686
    }
  ],
  "committed": true,
  "committed_count": 1,
  "commit_shas": ["0f140ab748ca6e0b215028760c909e9bb6ca26b8"],
  "commit_sha": "0f140ab748ca6e0b215028760c909e9bb6ca26b8",
  "commit_status": "committed and pushed",
  "head_at": "0f140ab",
  "head_note": "HEAD 0f140ab is the winner commit from this run (YOUTUBE_SEARCH_QUERIES[3] +50% NDR, ledger id 4686).",
  "last_runner_commit_sha": "0f140ab748ca6e0b215028760c909e9bb6ca26b8",
  "last_runner_commit_date": "2026-08-04",
  "committer_invoked": true,
  "committer_note": "1 winner auto-committed. AUTO_COMMIT_ENABLED=True. Config updated and pushed.",
  "winner_criterion": "status='completed' AND improvement_pct >= 0.20 AND avg weighted score did not drop AND both arms >= min_signals (tool_monitor >=3, others >=2). Per auto_research/evaluator.py and config.py.",
  "db_total_rows": 4689,
  "db_max_id": 4689,
  "db_max_timestamp": "2026-08-04T02:06:38.221167",
  "db_rows_dated_2026_08_04": 5,
  "anomaly": "Round 2 gemini_research errored with AttributeError: 'list' object has no attribute 'get' at mini_pipeline.py:324 in _parse_paid_signals. The Gemini API returned a list directly instead of a dict wrapping a 'signals' key. Gemini round 1 (id 4687) ran fine, so the issue is intermittent or query-specific. No ledger row for this error. New bug class -- not seen in prior logs.",
  "tool_monitor_structural_note": "Both tool_monitor slots hit insufficient_data again (0 and 2 signals vs min 3). The min-signals structural problem for this agent (MCP query class yielding <3 results/query) is unresolved. TOOL_SEARCH_QUERIES[1] (0 signals) and [5] (2 signals) are the recurrent low-yield slots.",
  "ledger_path": "auto_research/data/experiments.db",
  "runner_env_note": "Run used PYTHONPATH=/home/apexaipc/projects/research-agents; OLLAMA_BASE_URL=http://10.0.0.24:11434; OLLAMA_MODEL=qwen2.5:14b; .venv/bin/activate. AlienPC Ollama was responsive (qwen2.5:14b loaded)."
}
```

## Headline Numbers

| Metric | Value |
|---|---|
| Date / run timestamp | 2026-08-04, 02:01:30-02:06:56 CDT |
| Runner invocation | `python -m auto_research.runner --rounds 2` |
| **Experiments ran (persisted)** | **5 of 6 expected** |
| **Winners** | **1** -- youtube YOUTUBE_SEARCH_QUERIES[3] +50% NDR |
| **Committed** | **1** -- SHA `0f140ab7` |
| Committer invoked | yes |
| HEAD | `0f140ab` (this run's winner commit) |
| Ledger total rows | 4689 (5 added: ids 4685-4689) |
| Errors | 1 -- gemini_research round 2 AttributeError (no ledger row) |

## What Happened

**Round 1:**
- `tool_monitor` TOOL_SEARCH_QUERIES[5]: insufficient_data -- baseline only 2 signals (min 3). Recurring slot.
- `youtube` YOUTUBE_SEARCH_QUERIES[3]: **WINNER** -- NDR 66.7% -> 100.0% (+50%). Auto-committed `0f140ab7`.
- `gemini_research` GEMINI_RESEARCH_QUERIES[2]: completed, no winner -- guardrail triggered (avg score dropped 6.1->6.0 despite equal NDR 100%/100%).

**Round 2:**
- `tool_monitor` TOOL_SEARCH_QUERIES[1]: insufficient_data -- baseline 0 signals. Dead slot.
- `youtube` YOUTUBE_SEARCH_QUERIES[1]: completed, no winner -- guardrail triggered (avg score dropped 6.4->5.8 at NDR 100%/100%).
- `gemini_research` (round 2): **ERRORED** -- `AttributeError: 'list' object has no attribute 'get'` at `mini_pipeline.py:324`. Gemini API returned a list instead of a dict wrapping a `signals` key. No ledger row written.

## Winner Detail

```
Agent:     youtube
Param:     YOUTUBE_SEARCH_QUERIES[3]
Old query: "AI workflow automation with agent pipelines 2026"
New query: "AI driven workflow orchestration with autonomous agents 2026"
NDR:       66.7% -> 100.0% (+50%)
Signals:   10 baseline / 10 variant (both arms)
Commit:    0f140ab748ca6e0b215028760c909e9bb6ca26b8
```

## Action Items

1. **Investigate `_parse_paid_signals` AttributeError** (mini_pipeline.py:324): Gemini returned a `list` where a `dict` was expected. Either the response schema changed or a specific query triggers a different code path. Check what Gemini returned for the "workflow automation launches" query that errored.
2. **tool_monitor min-signals problem still open**: TOOL_SEARCH_QUERIES[1] (0 signals) and [5] (2 signals) are recurring insufficient_data slots. The MCP-query class for GitHub repo search is consistently below the min-3 bar. Triage: either lower min_signals for tool_monitor to 1, rewrite these specific queries, or add a fallback search strategy.
3. **youtube guardrail pattern**: Both round 2 youtube experiments hit NDR=100% on baseline and variant but failed the avg-score guardrail. The ceiling NDR (1.0) makes youtube NDR improvements rare in round 2 -- the guardrail is doing its job protecting score quality.
