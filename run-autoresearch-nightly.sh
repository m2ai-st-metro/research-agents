#!/bin/bash
# AutoResearch Nightly Runner
#
# Runs experiment loop against AlienPC's Ollama (RTX 5080).
# If AlienPC is unreachable, logs the miss and exits silently.
# Designed for cron — no interactive prompts, no hard failures.
#
# Schedule: 0 1 * * * (1 AM nightly, before research-agents at 5 AM)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/autoresearch-nightly.lock"
LOG_PREFIX="[AutoResearch]"
ALIENPC_OLLAMA="http://10.0.0.35:11434"
ROUNDS=20
NOTIFY_SCRIPT="/home/apexaipc/projects/claudeclaw/scripts/notify.sh"
MISSES_LOG="/home/apexaipc/logs/research-agents/autoresearch-misses.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $1"
}

notify() {
    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        "$NOTIFY_SCRIPT" "$1" 2>/dev/null || true
    fi
}

# --- Lock file ---
if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        log "Another run is active (PID $LOCK_PID). Exiting."
        exit 0
    else
        log "Stale lock file found. Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# --- Source shared env ---
if [[ -f "$HOME/.env.shared" ]]; then
    set -a
    source "$HOME/.env.shared"
    set +a
fi

log "Starting nightly AutoResearch run"

# --- Step 1: Ensure Ollama is running on AlienPC ---
# OLLAMA_HOST=0.0.0.0 is set persistently via setx on AlienPC.
# "OllamaServe" scheduled task runs ollama.exe serve.
# First check if already reachable; if not, try to start via schtasks.
OLLAMA_OK=$(curl -sf --connect-timeout 10 "${ALIENPC_OLLAMA}/api/tags" 2>/dev/null)
if [[ -z "$OLLAMA_OK" ]]; then
    log "Ollama not reachable. Attempting to start via schtasks..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes gaming-pc \
        "schtasks /Run /TN \"OllamaServe\"" 2>/dev/null || true
    sleep 15

    # Warm the model (first request triggers VRAM load, can take ~10s)
    curl -sf "${ALIENPC_OLLAMA}/api/generate" \
        -d '{"model":"qwen2.5:14b","prompt":"warmup","stream":false,"options":{"num_predict":1}}' \
        --connect-timeout 10 -m 60 >/dev/null 2>&1 || true
    sleep 2

    OLLAMA_OK=$(curl -sf --connect-timeout 10 "${ALIENPC_OLLAMA}/api/tags" 2>/dev/null)
fi

if [[ -z "$OLLAMA_OK" ]]; then
    log "AlienPC Ollama unreachable at ${ALIENPC_OLLAMA}. Skipping tonight."
    echo "$(date -Iseconds) AlienPC unreachable" >> "$MISSES_LOG"
    exit 0
fi

# Warm model if not already loaded (noop if already in VRAM, ~5s cold load)
log "Warming model..."
curl -sf "${ALIENPC_OLLAMA}/api/generate" \
    -d '{"model":"qwen2.5:14b","prompt":"warmup","stream":false,"options":{"num_predict":1}}' \
    --connect-timeout 10 -m 60 >/dev/null 2>&1 || true

log "AlienPC Ollama is available. Running ${ROUNDS} rounds."

# --- Step 2: Run experiments ---
cd "$SCRIPT_DIR"
source .venv/bin/activate

export OLLAMA_BASE_URL="$ALIENPC_OLLAMA"
export OLLAMA_MODEL="qwen2.5:14b"
export OLLAMA_TIMEOUT="300"

RESULT=$(python -m auto_research.runner --rounds "$ROUNDS" 2>&1)
EXIT_CODE=$?

echo "$RESULT"

if [[ $EXIT_CODE -ne 0 ]]; then
    log "Experiment run failed (exit $EXIT_CODE)"
    echo "$(date -Iseconds) Run failed exit=$EXIT_CODE" >> "$MISSES_LOG"
    exit 0
fi

# --- Step 3: Extract summary for notification ---
WINNER_COUNT=$(echo "$RESULT" | grep -c "WINNER" || echo "0")
EXPERIMENT_COUNT=$(echo "$RESULT" | grep -c "Experimenting on agent" || echo "0")

if [[ "$WINNER_COUNT" -gt 0 ]]; then
    WINNER_LINES=$(echo "$RESULT" | grep "WINNER" | sed 's/.*WINNER/WINNER/' | head -3)
    notify "AutoResearch nightly: ${EXPERIMENT_COUNT} experiments, ${WINNER_COUNT} winner(s)
${WINNER_LINES}"
fi

log "Complete. ${EXPERIMENT_COUNT} experiments, ${WINNER_COUNT} winners."
