#!/bin/bash
# AutoResearch Weekly Rollback Check
#
# Compares this week's NDR against last week's baseline.
# If NDR dropped >10%, auto-reverts committed changes.
# Schedule: 0 9 * * 1 (Monday 9 AM)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="[AutoResearch-Rollback]"
NOTIFY_SCRIPT="/home/apexaipc/projects/claudeclaw/scripts/notify.sh"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $1"
}

notify() {
    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        "$NOTIFY_SCRIPT" "$1" 2>/dev/null || true
    fi
}

# --- Source shared env ---
if [[ -f "$HOME/.env.shared" ]]; then
    set -a
    source "$HOME/.env.shared"
    set +a
fi

cd "$SCRIPT_DIR"
source .venv/bin/activate

log "Running weekly rollback check"

RESULT=$(python -c "
from auto_research.committer import check_weekly_rollback
from auto_research.ledger import init_db
import json

conn = init_db()

# Get this week's metrics from experiments
row = conn.execute('''
    SELECT AVG(variant_ndr) as current_ndr, AVG(variant_avg_score) as current_score
    FROM experiments
    WHERE committed = 1 AND timestamp >= date('now', '-7 days')
''').fetchone()

if row and row[0] is not None:
    rolled_back = check_weekly_rollback(
        conn,
        current_ndr=row[0],
        current_avg_score=row[1] or 0.0,
    )
    print(json.dumps({'rolled_back': rolled_back, 'ndr': row[0], 'score': row[1]}))
else:
    # No committed experiments this week — still save baseline from all experiments
    row2 = conn.execute('''
        SELECT AVG(baseline_ndr) as ndr, AVG(baseline_avg_score) as score
        FROM experiments
        WHERE status = 'completed' AND timestamp >= date('now', '-7 days')
    ''').fetchone()
    if row2 and row2[0] is not None:
        rolled_back = check_weekly_rollback(
            conn,
            current_ndr=row2[0],
            current_avg_score=row2[1] or 0.0,
        )
        print(json.dumps({'rolled_back': rolled_back, 'ndr': row2[0], 'score': row2[1]}))
    else:
        print(json.dumps({'rolled_back': False, 'ndr': None, 'message': 'no experiments this week'}))

conn.close()
" 2>&1)

EXIT_CODE=$?
echo "$RESULT"

if [[ $EXIT_CODE -ne 0 ]]; then
    log "Rollback check failed"
    exit 0
fi

# Notify if rollback happened
if echo "$RESULT" | grep -q '"rolled_back": true'; then
    notify "AutoResearch ROLLBACK triggered. NDR dropped >10% this week. Check logs."
fi

log "Rollback check complete"
