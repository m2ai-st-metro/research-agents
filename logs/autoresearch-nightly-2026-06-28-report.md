# AutoResearch Nightly Batch — Final Report (2026-06-28)

**Subtask 5 of 5** · Mission: "Run experiments and commit any winners."

> Status of this file: **PENDING FINALIZATION** until the run writes its
> `___EXIT_CODE___=` sentinel. All counts below are read directly from
> `auto_research/data/experiments.db`, never fabricated. The numbers are
> overwritten with the completed-run values the moment the run finishes.

## Headline (the three mission questions)

| Question | Answer (as of 02:20, run in progress) | Source |
|---|---|---|
| How many experiments ran? | **0 completed / ~6 expected** (1 of 6 started) | DB: 0 rows for `2026-06-28` |
| How many winners found? | **0** (none completed, so none can have won) | DB: `improvement_pct >= 0.20 AND status='completed'` = 0 |
| What was committed? | **Nothing** (`committed=1` rows for today = 0) | DB + git log |

## Run facts (verified)

- **Command:** `python -m auto_research.runner --rounds 2`
  (corrected venv: `.venv`, not `venv`), env `~/.env.shared` sourced.
- **Active run:** PID 1832807, started 02:15:01, detached (`setsid`).
  Log: `/tmp/autoresearch_run_20260628_021501_detached.log`.
- **Expected experiment count:** 3 agents (`tool_monitor`, `youtube`,
  `gemini_research`) × 2 rounds = **6 experiment comparisons**. Each = 1 baseline
  + 1 variant (`VARIANTS_PER_AGENT=1`).
- **Winner rule** (`config.py`): `improvement_pct >= 0.20` (IMPROVEMENT_THRESHOLD)
  AND guardrail pass AND min signals (`MIN_SIGNALS_PER_EXPERIMENT=2`,
  tool_monitor=3). `AUTO_COMMIT_ENABLED=True` → the runner auto-commits each winner
  inline (`commit_winner()`), sets `committed=1` + `commit_sha`.
- **History context:** the prior ~10 logged runs produced **0 winners**
  (improvement 0.0 or `insufficient_data`). Latest committed DB row is id 4598
  (tool_monitor, 2026-06-27, `insufficient_data`). A 0-winner result tonight would
  be the expected, not anomalous, outcome.
- **Note for the record:** the subtask-1 run died prematurely (session-scoped
  process killed at session end, log frozen at 02:11:40). Subtask 2 detected this
  and relaunched detached — that relaunched run is the one being reported here.

## Finalization query (run when sentinel appears)

```bash
DB=/home/apexaipc/projects/research-agents/auto_research/data/experiments.db
sqlite3 -header -column "$DB" "
  SELECT count(*) AS experiments,
         sum(improvement_pct>=0.20 AND status='completed') AS winners,
         sum(committed=1) AS committed
  FROM experiments WHERE timestamp LIKE '2026-06-28%';"
# Winner detail (IDs + SHAs):
sqlite3 -header -column "$DB" "
  SELECT id, agent, round(improvement_pct,3) AS impr, status, committed, commit_sha
  FROM experiments WHERE timestamp LIKE '2026-06-28%'
    AND improvement_pct>=0.20 AND status='completed' ORDER BY id;"
```

## Final numbers (filled on completion)

- Experiments run: _pending_
- Winners found: _pending_
- Committed (IDs + SHAs): _pending_
