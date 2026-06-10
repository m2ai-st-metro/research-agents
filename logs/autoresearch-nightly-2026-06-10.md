---
agent: claude-code
date: 2026-06-10
runner_invocation: python -m auto_research.runner --rounds 2
runtime_window: ~02:01–02:11 local (terminated early, Round 1 of 2)
status: INCOMPLETE — run did not finish
---

# AutoResearch Nightly Batch — 2026-06-10

## Headline Numbers

| Metric | Value |
|---|---|
| Rounds requested | 2 |
| Experiments logged | **1** (id 4585 — of ~8 expected: 4 agents × 2 rounds) |
| Experiments started but killed mid-run | **1** (youtube) |
| Valid comparisons | **0** |
| Winners (NDR Δ ≥ 20%, guardrail-passed) | **0** |
| Winners committed | **0** |
| Commit hashes | **none** |
| Run completion | **INCOMPLETE** (early termination at ~02:11, Round 1 of 2) |

## Per-Experiment Outcomes

| Exp ID | Round | Agent | Param / Slot | Result | NDR Δ | Committed |
|---|---|---|---|---|---|---|
| 4585 | 1 | tool_monitor | `TOOL_SEARCH_QUERIES[5]` | insufficient_data (baseline 1 signal, min 3) | 0% (NDR 100% → 0%) | 0 |
| — | 1 | youtube | `MCP server creation guide 2026` slot | terminated mid-run, no row recorded | — | — |

Detail (id 4585): logged `2026-06-10`. Baseline query `MCP SDK typescript python client
binding` returned **1 relevant signal** (NDR 100%, avg score 7.175) — below the `tool_monitor`
`min_signals=3` validity gate. Variant query `MCP client-server bindings for Python and TypeScript`
hit a **GitHub API 403 rate-limit** → 0 signals, NDR 0%. Per `evaluator.compare`, the winner gate
requires `improvement_pct ≥ 0.20` **AND** avg-score not dropped **AND** `is_valid` (baseline and
variant each ≥ min_signals). Row 4585 failed `is_valid` first (reason: baseline 1 signal < min 3),
so `improvement_pct=0.0`, `status=insufficient_data`, `is_winner=False`, `committed=0`,
`commit_sha=NULL`. The comparison was confounded by the rate-limit on the variant rather than a
genuine baseline-vs-variant test.

Only this single `tool_monitor` experiment reached `log_experiment`. The `youtube` experiment was
killed mid-execution at ~02:11 before any result was persisted; Round-1 `gemini_research` /
`reddit` (and all of Round 2) never executed.

## Committed Artefacts

- **None.** 0 winners → committer correctly took no action.
- No autoresearch commits dated 2026-06-10. `src/research_agents/config.py` untouched; no `git push`.
- Working tree shows only `auto_research/data/experiments.db` modified (the one ledger row written).

## Committer Verification

- Tonight's winners (`timestamp LIKE '2026-06-10%' AND status='completed' AND improvement_pct ≥ 0.20`):
  **0 rows**.
- Tonight's status breakdown: `insufficient_data: 1`. Rows with `committed=1` tonight: **0**. Rows with
  `commit_sha` set tonight: **0**.
- `auto_research/committer.py` is **library-only** — no `__main__`, no argparse. There is no standalone
  committer CLI; `commit_winner(...)` is called inline by `runner.py` (~line 226/233) during the run,
  gated by `comparison.is_winner AND AUTO_COMMIT_ENABLED`. The run already passed that path during the
  runner invocation and committed nothing because the single experiment was a non-winner. The mission's
  suggested `python -m auto_research.committer` does not exist by design.
- Canonical ledger is `auto_research/data/experiments.db` (1.9 MB, 4585 experiments). The root-level
  `data/experiments.db` and `data/autoresearch.db` are empty 0-byte stubs and are **not** used.

## Operational Flags

- ⚠ **Run terminated early — 6th consecutive incomplete nightly** (2026-06-05: 1 exp, 2026-06-06: 1 exp,
  2026-06-08: 2 exp, 2026-06-09: 1 exp, now 2026-06-10: 1 exp + 1 killed). Same failure mode: the runner
  is launched as a background task that dies when the launching session ends. ~7 of ~8 expected
  comparisons never ran. **Fix:** run the batch as a foreground/tracked task (or via the cron
  `run-agents.sh` wrapper) so it isn't killed mid-run.
- ⚠ **`tool_monitor` insufficient_data streak continues.** Id 4585 (and 4584 on 06-09, 4582–4583 on
  06-08, 4581 on 06-06) all logged `insufficient_data` / 0.0 — queries are not yielding the ≥3 signals
  needed for a valid comparison. Tonight the variant additionally hit a **GitHub API 403 rate-limit**.
  The harness is being starved of valid data; the GitHub-signal yield path (and rate-limit handling)
  is worth investigating separately.
- ⚠ **Historical uncommitted "would-be-winner" backlog (~136 rows, all-time).** `get_winners()` returns
  136 rows with `improvement_pct ≥ 0.20 AND committed=0`, but **every one is stale historical**
  (dated 2026-03-20 → 2026-05-27 — **none** from tonight). These were **NOT** mass-committed: (1) they
  are out of scope for this batch; (2) committing rewrites `src/research_agents/config.py` and pushes to
  the shared `m2ai-st-metro` GitHub remote — irreversible, no authorization for these rows; (3) their
  `baseline_value` query strings are months old, so most no longer exist in current `config.py` and
  `_replace_query_in_config` would warn/fail. Root cause: the per-run committer never sweeps historical
  winners, and `AUTO_COMMIT_ENABLED` only flipped on 2026-04-19. Any future reconciliation committer must
  re-apply the avg-weighted-score guardrail (`not score_dropped`) and skip retired-agent rows — never
  filter on `improvement_pct` alone. **Flagged for a separate human decision; untouched by this run.**

## Hive-Mind Sink Note

The CMD HiveMind (`cmd_hivemind` MCP tool) is a **query-only** read interface over the cross-agent
activity log; it is auto-populated by the CMD orchestrator during real CMD missions and has no manual
write method. The durable, writable sink for AutoResearch nightly batches is this markdown nightly-log
series (`logs/autoresearch-nightly-YYYY-MM-DD.md`), consistent with 2026-04-29, 2026-06-05, 2026-06-06,
2026-06-08, and 2026-06-09. This file is that log entry. A one-line pointer is also appended to the
vault running log (`~/vault/log.md`).

## Bottom Line

- **Experiments run:** 1 logged (tool_monitor, id 4585, insufficient_data) + 1 killed mid-run (youtube);
  run terminated early at ~02:11 (Round 1 of 2; ~7 of ~8 expected never ran).
- **Winners found:** 0.
- **Committed:** none (no config change, no push; repo clean apart from the one ledger row).
