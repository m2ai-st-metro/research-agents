---
mission_id: 18be91de-6f1e-4e86-baea-5cf0a7af3842
agent: content
date: 2026-04-29
runner_invocation: python -m auto_research.runner --rounds 2
runtime_window: 02:16 - 02:25 local
---

# AutoResearch Nightly Batch — 2026-04-29

## Headline Numbers

| Metric | Value |
|---|---|
| Rounds | 2 |
| Experiments run | 8 (4 agents × 2 rounds) |
| Winners | 1 |
| Winners committed | 1 |
| Errors (non-fatal) | 4 (Perplexity 401, agent skipped) |

## Per-Experiment Outcomes

| Exp ID | Round | Agent | Result | NDR Δ |
|---|---|---|---|---|
| 2827 | 1 | gemini_research | guardrail (NDR flat, score dropped) | 0% |
| 2828 | 1 | youtube | guardrail | -33% |
| 2829 | 1 | perplexity | insufficient_data (401) | n/a |
| 2830 | 1 | gemini_research | no NDR change | 0% |
| 2831 | 2 | tool_monitor | **WINNER, committed** | **+50%** |
| 2832 | 2 | youtube | guardrail | -33% |
| 2833 | 2 | perplexity | insufficient_data (401) | n/a |
| 2834 | 2 | gemini_research | no NDR change | 0% |

Round 1 `tool_monitor`: insufficient signals (1 < 3 min). Round 2 promoted to winner.

## Committed Artefacts

- **Commit:** `7d6151beacadc4d2e876d59639f9ea71cec1ee78`
- **Author:** Matthew Snow (co-author: AutoResearch <noreply@autoresearch.local>)
- **Timestamp:** 2026-04-29 02:21:51 -0500 (during runner execution)
- **Branch:** master (up to date with origin/master)
- **Message:** `auto-research: update TOOL_SEARCH_QUERIES query (+50% NDR)`
- **Change:** `tool_monitor` agent — TOOL_SEARCH_QUERIES query
  - Old: `multi-agent coordination framework implementation`
  - New: `multi-agent workflow management system implementation`
- **Improvement:** +50.0% non-dismiss rate (threshold was 20%)

## Committer Verify (Subtask 2)

- Pending uncommitted winners ledger query: **0 rows** (`improvement_pct >= 0.15 AND committed = 0 AND rolled_back = 0` for today)
- `auto_research.committer` is library-only (no CLI entrypoint) — invoked internally by `runner.py`. Manual re-invocation N/A.
- No additional commits required.

## Operational Flags

- **Perplexity 401 Unauthorized** on experiments 2829 + 2833 — pre-existing API key issue, NOT introduced by this run. Agent skipped, run did not fail. Action: rotate `PERPLEXITY_API_KEY` in `~/.env.shared`.
- **Working tree dirty** at runtime (`CLAUDE.md`, `mutator.py`, `runner.py`, etc.) — unrelated to this run, did not affect execution.

## Mission Lineage

- Mission start: ts 1777446031
- Subtask 1 (run experiments): 607s on `content` (retried once)
- Subtask 2 (committer verify): 106s on `claude-code`, no-op
- Subtask 3 (this report): in flight on `claude-code`
