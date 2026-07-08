# AutoResearch Nightly — 2026-07-08 (REPORT-TIME SNAPSHOT, run still executing)

**Status at 02:14 CT: batch STILL RUNNING (detached), 0 experiments completed yet.**
This file is a snapshot at report time, NOT the final tally. Read the ledger for finals (query below).

## Verified numbers at report time
| Metric | Value | Source |
|---|---|---|
| Experiments completed tonight (2026-07-08) | **0** | ledger: 0 rows dated 2026-07-08 |
| Winners tonight | **0** | derived: no completed experiments |
| Commits tonight | **0** | git HEAD unchanged at `5d3afc7` |
| Ledger all-time total | 4644 | `SELECT COUNT(*)` |
| Ledger all-time committed | 239 | `SELECT SUM(committed)` |
| Newest ledger row | `2026-07-07T02:29:36` (id 4644) | prior night's run |

## Run state
- Command: `python -u -m auto_research.runner --rounds 2` (pid 2975173, detached via `setsid` in subtask 2 — survives session teardown).
- Started 02:07 CT; at 02:14 (~6.5 min in) still on **Round 1/2, agent `tool_monitor`, baseline phase of experiment #1** (of ~10 expected: ~5 Ollama agents × 2 rounds).
- **Root cause of slowness:** AlienPC GPU (`10.0.0.24`, ~7s/assessment) is OFF tonight; running CPU fallback via `localhost:11434` (~30-40s/assessment). As of 02:13:34 Ollama requests began **timing out with retry backoff** — degrading further.
- A ledger row persists only when an experiment fully completes, so max id stays 4644 until experiment #1 finishes.

## Commit mechanism (for when winners land)
`AUTO_COMMIT_ENABLED=True` → runner auto-commits each winner inline via `commit_winner()` (edits `config.py` → git add/commit/push → marks ledger `committed=1`). Winner = variant beats baseline NDR by ≥20% with sufficient signals (tool_monitor ≥3, else ≥2). No manual committer step needed unless a push fails.

## NOT committed (deliberate)
`get_winners()` surfaces ~20 old "uncommitted winners" — March–May rows, mostly from **retired agents** (arxiv, domain_watch) predating `AUTO_COMMIT_ENABLED` (2026-04-19). These are a trap; committing stale mutations against since-changed config queries would be wrong. Left untouched.

## Read-back query for final numbers (once the detached run finishes overnight)
```bash
sqlite3 /home/apexaipc/projects/research-agents/auto_research/data/experiments.db \
  "SELECT COUNT(*) AS ran,
          SUM(CASE WHEN improvement_pct>=0.20 THEN 1 ELSE 0 END) AS winners,
          SUM(committed) AS committed
   FROM experiments WHERE timestamp LIKE '2026-07-08%';"
git -C /home/apexaipc/projects/research-agents log --oneline --since='2026-07-08 00:00' | grep auto-research
```
