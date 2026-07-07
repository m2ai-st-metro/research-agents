---
agent: claude-code
date: 2026-07-07
runner_invocation: python -m auto_research.runner --rounds 2
runtime_window: 2026-07-07 nightly batch. Subtask 1 launched the run as a detached background job inside a subagent session; that job was reaped at 02:19:29 when the subagent session ended (same session-scoped-background-task failure class as 07-04/05/06), persisting partial rows 4632-4638. Subtask 2 found no usable run on entry, confirmed Ollama healthy, and completed a clean FOREGROUND (session-tracked, un-orphanable) `--rounds 2` run 02:22:48 -> 02:29:41 (exit 0), persisting rows 4639-4644. This is the authoritative run for the mission.
status: COMPLETE (clean foreground run, exit 0) -- 6 experiments, 0 winners, 0 commits.
---

# AutoResearch Nightly Batch — 2026-07-07

## Hive-Mind Payload (machine-readable)

```json
{
  "sink": "logs/autoresearch-nightly-2026-07-07.md",
  "date": "2026-07-07",
  "runner_invocation": "python -m auto_research.runner --rounds 2",
  "run_status": "complete",
  "run_exit_code": 0,
  "run_window": "2026-07-07T02:22:48 -> 02:29:41 (foreground, session-tracked)",
  "rounds_requested": 2,
  "rounds_completed": 2,
  "agents_per_round": ["tool_monitor", "youtube", "gemini_research"],
  "experiments_ran": 6,
  "experiment_ledger_ids": [4639, 4640, 4641, 4642, 4643, 4644],
  "experiment_results": [
    {"ledger_id": 4639, "round": 1, "agent": "tool_monitor", "param_name": "TOOL_SEARCH_QUERIES[5]", "baseline_signals": 1, "variant_signals": 0, "min_signals": 3, "baseline_ndr": 1.0, "variant_ndr": 0.0, "improvement_pct": 0.0, "status": "insufficient_data", "is_winner": false, "notes": "Baseline has only 1 signals (min: 3)"},
    {"ledger_id": 4640, "round": 1, "agent": "youtube", "param_name": "YOUTUBE_SEARCH_QUERIES[3]", "baseline_signals": 10, "variant_signals": 10, "baseline_ndr": 1.0, "variant_ndr": 0.667, "baseline_avg_score": 5.98, "variant_avg_score": 5.42, "improvement_pct": -0.333, "status": "completed", "is_winner": false, "notes": "NDR dropped and avg score dropped (6.0 -> 5.4). Guardrail triggered."},
    {"ledger_id": 4641, "round": 1, "agent": "gemini_research", "param_name": "GEMINI_RESEARCH_QUERIES[0]", "baseline_signals": 5, "variant_signals": 4, "baseline_ndr": 1.0, "variant_ndr": 1.0, "improvement_pct": 0.0, "status": "completed", "is_winner": false, "notes": "No change in non-dismiss rate"},
    {"ledger_id": 4642, "round": 2, "agent": "tool_monitor", "param_name": "TOOL_SEARCH_QUERIES[5]", "baseline_signals": 1, "variant_signals": 0, "min_signals": 3, "baseline_ndr": 0.667, "variant_ndr": 0.0, "improvement_pct": 0.0, "status": "insufficient_data", "is_winner": false, "notes": "Baseline has only 1 signals (min: 3)"},
    {"ledger_id": 4643, "round": 2, "agent": "youtube", "param_name": "YOUTUBE_SEARCH_QUERIES[3]", "baseline_signals": 10, "variant_signals": 10, "baseline_ndr": 1.0, "variant_ndr": 0.667, "baseline_avg_score": 5.98, "variant_avg_score": 5.43, "improvement_pct": -0.333, "status": "completed", "is_winner": false, "notes": "NDR dropped and avg score dropped (6.0 -> 5.4). Guardrail triggered."},
    {"ledger_id": 4644, "round": 2, "agent": "gemini_research", "param_name": "GEMINI_RESEARCH_QUERIES[2]", "baseline_signals": 5, "variant_signals": 5, "baseline_ndr": 1.0, "variant_ndr": 1.0, "improvement_pct": 0.0, "status": "completed", "is_winner": false, "notes": "No change in non-dismiss rate"}
  ],
  "winners_count": 0,
  "winners": [],
  "committed": false,
  "committed_count": 0,
  "committed_artifact_names": [],
  "committed_files": [],
  "commit_shas": [],
  "commit_hash": null,
  "commit_sha": "none",
  "commit_status": "no commit",
  "committer_invoked": true,
  "committer_note": "Subtask 2 ran a verified sweep, not a blind commit: zero rows from this run meet any winner criterion, so commit_winner() had no eligible input and correctly took no action. The runner's inline auto-commit path (AUTO_COMMIT_ENABLED=true) also never fired -- no variant cleared the +20% NDR gate. All 6 rows have committed=0 and commit_sha=null.",
  "winner_criterion": "is_winner = status='completed' AND improvement_pct >= IMPROVEMENT_THRESHOLD (0.20) AND avg weighted score did not drop AND valid (baseline & variant each >= min_signals: tool_monitor >=3, others >=2). Per auto_research/evaluator.py and config.py.",
  "auto_commit_enabled": true,
  "improvement_threshold": 0.20,
  "notify_fired": false,
  "notify_note": "Runner notify path fires only on a non-zero winner count; 0 winners means silent, the expected outcome.",
  "head_at": "5d2dc0f",
  "head_note": "WIP: auto-snapshot 2026-07-07 02:30:01 -- git-wip-snapshot cron, NOT a runner commit. Working tree clean; auto_research/config.py diff is empty (no query mutation applied).",
  "last_runner_commit_sha": "735058b",
  "last_runner_commit_date": "2026-07-03",
  "last_runner_commit_note": "auto-research: update YOUTUBE_SEARCH_QUERIES query (+100% NDR)",
  "ledger_path": "auto_research/data/experiments.db",
  "db_total_rows": 4644,
  "db_max_id": 4644,
  "db_max_timestamp": "2026-07-07T02:29:36.031080",
  "db_rows_dated_2026_07_07": 13,
  "db_rows_from_this_run": 6,
  "db_rows_from_aborted_attempts": 7,
  "aborted_attempt_ids": [4632, 4633, 4634, 4635, 4636, 4637, 4638],
  "aborted_attempt_note": "Rows 4632-4638 (02:03:04 -> 02:19:18) are partial output from subtask 1's detached background run plus earlier relaunch attempts, reaped at session end. None are winners; none committed. Counted separately from the authoritative 6-experiment run.",
  "legacy_uncommitted_winners_all_time": 136,
  "legacy_uncommitted_winners_today": 0,
  "legacy_winners_note": "The exact ledger.get_winners() query (ledger.py:146: status='completed' AND improvement_pct >= 0.15 AND committed=0 AND rolled_back=0) matches 136 stale rows spanning 2026-03-20..2026-05-27, mostly the retired arxiv agent, predating auto-commit enablement (2026-04-19). ZERO of them are from tonight. Any future committer sweep of get_winners() MUST filter by timestamp or it will apply months-old query mutations without fresh evidence. Separate triage decision for Matthew (bulk-expire vs re-validate).",
  "infra": {
    "ollama_base_url": "http://10.0.0.24:11434",
    "ollama_model": "qwen2.5:14b",
    "ollama_health": "reachable, model loaded",
    "ollama_timeout_s": 300,
    "note": "AlienPC GPU (RTX 5080) healthy this run -- experiments ran on GPU, not the ProBook CPU fallback."
  }
}
```

## Headline Numbers

| Metric | Value |
|---|---|
| Date / run window | 2026-07-07, 02:22:48 -> 02:29:41 CDT |
| Runner invocation | `python -m auto_research.runner --rounds 2` |
| Run status | **COMPLETE** (exit 0, foreground/session-tracked) |
| Rounds requested / completed | 2 / 2 |
| **Experiments ran** | **6** (2 rounds x 3 agents: tool_monitor, youtube, gemini_research; ledger ids 4639-4644) |
| **Winners** | **0** |
| **Committed** | **nothing** — commit SHA `none` |
| Committer invoked | on 0 winners (verified-empty winner set; correct no-op, not a skipped step) |
| HEAD | `5d2dc0f` (wip-snapshot 02:30, not a runner commit); working tree clean, `config.py` diff empty |
| Last real auto-research commit | `735058b` (2026-07-03, youtube query, +100% NDR) — unchanged |
| Ledger total rows | 4644 (6 added by this run; +7 more from aborted attempts = 13 rows dated 2026-07-07) |
| Legacy uncommitted "winners" left untouched | 136 (all-time; **0** from tonight) |
| Ollama | AlienPC GPU `qwen2.5:14b` healthy — ran on GPU, not CPU fallback |

## What Happened

- **The batch ran clean this time.** Unlike the previous three nights (07-04/05/06), the completed `--rounds 2` run finished normally: exit 0, 02:22:48 -> 02:29:41, all 6 experiments persisted (ledger ids 4639-4644). Ollama on AlienPC (`qwen2.5:14b`, RTX 5080) was healthy, so experiments ran on GPU rather than the slow ProBook CPU fallback.
- **6 experiments, 0 winners.** No variant cleared the `IMPROVEMENT_THRESHOLD` (+20% NDR with no avg-score drop and valid signal counts):
  - `tool_monitor` (4639, 4642) — `TOOL_SEARCH_QUERIES[5]`: baseline only 1 signal (min 3) -> **insufficient_data** both rounds.
  - `youtube` (4640, 4643) — `YOUTUBE_SEARCH_QUERIES[3]`: variant NDR dropped 100% -> 66.7% AND avg score dropped 6.0 -> 5.4 -> **guardrail rejected** both rounds (working exactly as designed: a worse variant is rejected, not committed).
  - `gemini_research` (4641, 4644) — `GEMINI_RESEARCH_QUERIES[0]`/`[2]`: NDR unchanged 100% <-> 100% -> **no change**, no winner.
- **Winners: 0. Committed: nothing.** Auto-commit is inline (`AUTO_COMMIT_ENABLED=true`), so any winner would have committed itself; none qualified, so `commit_winner()` had no eligible input and the committer correctly took no action. HEAD is `5d2dc0f` (a 02:30 wip-snapshot cron commit, not a runner commit); `config.py` has an empty diff and the working tree is clean — nothing half-applied. Last real auto-research commit remains `735058b` (2026-07-03).
- **Launch-lifecycle note (not a result, but worth carrying forward):** subtask 1 launched the run as a detached background job *inside a subagent session*, and it was reaped at 02:19:29 when that session ended — the same session-scoped-background-task failure class logged 07-04/05/06. Subtask 2 recovered by running the batch in the **foreground** (session-tracked, un-orphanable). The aborted attempts left 7 partial rows (4632-4638), none winners, none committed; they are counted separately from the authoritative 6-experiment run above.
- **Legacy trap re-verified:** the loose `get_winners()` query (threshold 0.15, `ledger.py:146`) still matches **136 stale uncommitted rows** from 2026-03..2026-05, mostly the retired `arxiv` agent, all predating auto-commit enablement. **Zero** are from tonight. Any future committer sweep MUST filter by timestamp. Backlog triage (bulk-expire vs re-validate) is a separate decision for Matthew.

## Hive-Mind Sink Note

This dated nightly log **is** the hive-mind sink for the AutoResearch batch (same convention as 06-06 through 07-06). The CMD HiveMind (`ST Metro CMD MCP` / `cmd_hivemind`) is a query-only read interface, exposes no write tool, and requires an OAuth authorization that is not available in this non-interactive session, so structured batch results are written here as the machine-readable payload above for downstream readers.

## Bottom Line

- **Experiments ran:** 6 (2 rounds x 3 agents; clean foreground run, exit 0). First fully-complete nightly batch after three consecutive session-teardown failures.
- **Winners found:** 0 (2 insufficient_data, 2 guardrail-rejected, 2 no-change — the guardrail worked as designed).
- **Committed:** nothing — commit SHA `none`, HEAD unchanged at `5d2dc0f` (a wip-snapshot); last real runner commit is still `735058b` (2026-07-03).
- **Operational lesson (carried into memory):** the nightly batch must run foreground/session-tracked, never as a detached background job inside a subagent session, or it gets reaped at session teardown — exactly what killed subtask 1's launch tonight and the 07-04/05/06 runs.
