#!/bin/bash
# Research Agents Runner
#
# Runs specified research agents with lock file protection.
#
# Usage:
#   ./run-agents.sh arxiv tool-monitor    # Run specific agents
#   ./run-agents.sh --dry-run arxiv       # Dry run mode
#   ./run-agents.sh idea-surfacer         # Run single agent

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/research-agents.lock"
LOG_PREFIX="[ResearchAgents]"
DRY_RUN=""
AGENTS=()

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN="--dry-run"
    else
        AGENTS+=("$arg")
    fi
done

if [[ ${#AGENTS[@]} -eq 0 ]]; then
    echo "$LOG_PREFIX No agents specified. Usage: ./run-agents.sh [--dry-run] <agent1> [agent2] ..."
    echo "$LOG_PREFIX Available agents: tool-monitor, rss, youtube, perplexity, chatgpt, gemini-research, reddit, trend-analyzer, idea-surfacer"
    exit 1
fi

# --- Lock file to prevent overlapping runs ---
MAX_LOCK_AGE_SECONDS=3600  # 1 hour — no single run should take longer

if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    LOCK_AGE=0
    if [[ -f "$LOCK_FILE" ]]; then
        LOCK_CREATED=$(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        LOCK_AGE=$(( NOW - LOCK_CREATED ))
    fi

    if [[ "$LOCK_AGE" -ge "$MAX_LOCK_AGE_SECONDS" ]]; then
        echo "$LOG_PREFIX Lock file is ${LOCK_AGE}s old (max ${MAX_LOCK_AGE_SECONDS}s). Force-removing stale lock (PID $LOCK_PID)."
        rm -f "$LOCK_FILE"
    elif [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "$LOG_PREFIX Another run is active (PID $LOCK_PID, age ${LOCK_AGE}s). Exiting."
        exit 1
    else
        echo "$LOG_PREFIX Stale lock file found (PID $LOCK_PID not running). Removing."
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# --- Source shared env for cron context ---
if [[ -f "$HOME/.env.shared" ]]; then
    set -a
    source "$HOME/.env.shared"
    set +a
fi

echo "============================================================"
echo "  $LOG_PREFIX Agent Run"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Agents: ${AGENTS[*]}"
echo "  Dry run: ${DRY_RUN:-false}"
echo "============================================================"

# --- Activate venv ---
cd "$SCRIPT_DIR"
source .venv/bin/activate

# --- Run each agent ---
for agent in "${AGENTS[@]}"; do
    echo ""
    echo ">>> Running: $agent"
    echo "------------------------------------------------------------"
    research-agents run "$agent" $DRY_RUN || {
        echo "$LOG_PREFIX WARNING: Agent '$agent' failed with exit code $?"
        # Continue to next agent instead of aborting
    }
done

echo ""
echo "$LOG_PREFIX Complete at $(date '+%Y-%m-%d %H:%M:%S')"
