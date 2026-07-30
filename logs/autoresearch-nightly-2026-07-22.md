---
agent: claude-code (mission subtask 3/4)
date: 2026-07-22
runner_invocation: python -m auto_research.runner --rounds 2
runtime_window: 2026-07-22 nightly batch, started ~02:03 CDT. Runner launched in background (task bog62ibc4), restarted once (bxaokenv0). Only 1 experiment persisted to the ledger; no runner process alive at log time (~mission subtask 3). Run is INCOMPLETE.
status: INCOMPLETE (1/6 expected experiments persisted) -- 0 winners, 0 commits.
---

# AutoResearch Nightly Batch -- 2026-07-22

## Hive-Mind Payload (machine-readable)

```json
{
  "sink": "logs/autoresearch-nightly-2026-07-22.md",
  "date": "2026-07-22",
  "run_timestamp": "2026-07-22T02:03:52.083393",
  "runner_invocation": "python -m auto_research.runner --rounds 2",
  "rounds_requested": 2,
  "rounds_completed": "partial (1 of ~6 expected experiments persisted; runner process not alive at verification time)",
  "experiments_ran": 1,
  "experiments_expected_full_run": 6,
  "experiments_missing": 5,
  "experiment_results": [
    {
      "ledger_id": 4680,
      "timestamp": "2026-07-22T02:03:52.083393",
      "agent": "tool_monitor",
      "param_name": "TOOL_SEARCH_QUERIES[5]",
      "baseline_value": "MCP SDK typescript python client binding",
      "variant_value": "MCP SDK typescript python client binding API",
      "baseline_signals": 1,
      "variant_signals": 0,
      "baseline_ndr": 0.0,
      "variant_ndr": 0.0,
      "improvement_pct": 0.0,
      "status": "insufficient_data",
      "notes": "Baseline has only 1 signals (min: 3)",
      "is_winner": false
    }
  ],
  "winners_count": 0,
  "winners": [],
  "committed": false,
  "committed_count": 0,
  "commit_shas": [],
  "commit_sha": "none",
  "commit_status": "no commit this run",
  "head_at": "3b0b22b",
  "head_note": "HEAD 3b0b22b is the 2026-07-20 batch's winner commit (GEMINI_RESEARCH_QUERIES +50% NDR, ledger id 4674), NOT from this run. Sibling 5c95c38 (YOUTUBE_SEARCH_QUERIES +50% NDR, id 4673) also 07-20. No new commits on 07-22.",
  "last_runner_commit_sha": "3b0b22b",
  "last_runner_commit_date": "2026-07-20",
  "committer_invoked": false,
  "committer_note": "0 winners from this run; nothing eligible to commit. HEAD unchanged at 3b0b22b.",
  "winner_criterion": "status='completed' AND improvement_pct >= 0.20 AND avg weighted score did not drop AND both arms >= min_signals (tool_monitor >=3, others >=2). Per auto_research/evaluator.py and config.py.",
  "db_total_rows": 4680,
  "db_max_id": 4680,
  "db_max_timestamp": "2026-07-22T02:03:52.083393",
  "db_rows_dated_2026_07_22": 1,
  "legacy_uncommitted_winners_left_untouched": 137,
  "legacy_winners_note": "137 rows match completed + improvement>=20% + committed=0 (was 136 in prior logs; mostly stale 2026-03..2026-05). Not touched. Still require timestamp-filtered sweeps before any committer action.",
  "anomaly": "Runner was launched in background (task bog62ibc4), stalled, and was restarted (bxaokenv0). At log time no auto_research.runner process is alive and only 1 of ~6 expected experiments persisted (id 4680, 02:03:52). The restarted runner appears to have died after its first experiment without completing rounds. Rows persist per-experiment, so the single result is safe; the remaining ~5 experiments simply never ran.",
  "root_cause_class": "runner-died-mid-run (restart did not complete)",
  "recurring_pattern_note": "The one experiment that did run reproduces the known tool_monitor min-signals failure (baseline 1 signal vs min 3, MCP query class). The NDR-ceiling + min-signals structural problem flagged in the 07-10 log remains untriaged; 07-20 did produce 2 winners (youtube, gemini_research +50% NDR), so ceiling is not absolute.",
  "ledger_path": "auto_research/data/experiments.db"
}
```

## Headline Numbers

| Metric | Value |
|---|---|
| Date / run timestamp | 2026-07-22, first (only) experiment 02:03:52 CDT |
| Runner invocation | `python -m auto_research.runner --rounds 2` |
| **Experiments ran (persisted)** | **1 of ~6 expected -- run INCOMPLETE** |
| **Winners** | **0** |
| **Committed** | **nothing** -- commit SHA `none` |
| Committer invoked | no (0 winners) |
| HEAD | unchanged at `3b0b22b` (07-20 winner commit, prior batch) |
| Ledger total rows | 4680 (1 added by this run: id 4680) |
| Legacy uncommitted "winners" untouched | 137 (stale, mostly 2026-03..2026-05) |

## What Happened

- Subtask 1 launched the runner in background (task `bog62ibc4`); it stalled and was restarted (`bxaokenv0`).
- The restarted runner persisted exactly one experiment (ledger id 4680, `tool_monitor` `TOOL_SEARCH_QUERIES[5]`): baseline 1 signal vs min 3 -- `insufficient_data`, not a winner.
- No `auto_research.runner` process was alive at verification time and no further rows landed, so the remaining ~5 experiments of the 2-round batch never ran.
- **Winners: 0. Committed: nothing.** HEAD remains `3b0b22b`; the two recent winner commits (`3b0b22b`, `5c95c38`, both +50% NDR) belong to the 2026-07-20 batch, not tonight.

## Bottom Line

- **Experiments ran:** 1 (of ~6 expected; runner died mid-run after restart).
- **Winners found:** 0 (the lone experiment hit the tool_monitor min-signals gate).
- **Committed:** nothing; HEAD unchanged at `3b0b22b` (last real winner commits are from 2026-07-20).
- **Action:** investigate why the restarted runner died after one experiment (subtask 4 / next batch); the min-signals / NDR-ceiling triage from the 07-10 log is still open.
