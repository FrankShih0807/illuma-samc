#!/bin/bash
# Watch TODO.md for new unchecked items and launch child Claude automatically.
# Usage: ./scripts/watch_todo.sh
# Requires: fswatch (brew install fswatch)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TODO_FILE="$PROJECT_DIR/TODO.md"
LOCK_FILE="$PROJECT_DIR/.claude-child.lock"
LOG_FILE="$PROJECT_DIR/logs/child-claude.log"

mkdir -p "$PROJECT_DIR/logs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[watcher]${NC} $(date '+%H:%M:%S') $1"; }
warn() { echo -e "${YELLOW}[watcher]${NC} $(date '+%H:%M:%S') $1"; }

count_unchecked() {
    grep -c '^\- \[ \]' "$TODO_FILE" 2>/dev/null || echo 0
}

# Robust lock check: verify PID is actually alive, clean stale locks
check_and_clean_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # running
        else
            log "Stale lock found (PID $pid dead). Cleaning up."
            rm -f "$LOCK_FILE"
            return 1  # was stale, now cleaned
        fi
    fi
    return 1  # no lock
}

launch_claude() {
    local unchecked
    unchecked=$(count_unchecked)
    if [ "$unchecked" -eq 0 ]; then
        log "No unchecked items in TODO.md, skipping."
        return
    fi

    if check_and_clean_lock; then
        warn "Child Claude is already running (PID $(cat "$LOCK_FILE")), skipping."
        return
    fi

    log "Found $unchecked unchecked TODO items. Launching child Claude..."

    cd "$PROJECT_DIR"

    # Pull latest and apply migrations before starting
    git pull --rebase origin main 2>/dev/null || true
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate illuma-samc 2>/dev/null

    claude --allowedTools "Bash,Edit,Read,Write,Glob,Grep" \
        -p "You are an L3 Worker. Read ../AGENT_PLAYBOOK.md Section 5 for your start/end checklists. Read CLAUDE.md and WORKLOG.md (last 3 entries). Work through TODO.md IN ORDER — complete all items in a step before moving on. If stuck, create BLOCKED.md." \
        >> "$LOG_FILE" 2>&1 &

    local child_pid=$!
    echo "$child_pid" > "$LOCK_FILE"
    log "Child Claude launched (PID $child_pid). Logs: $LOG_FILE"

    # When child exits: clean up lock, auto-resume if items remain
    (
        wait "$child_pid" 2>/dev/null
        rm -f "$LOCK_FILE"
        log "Child Claude finished (exit)."

        # Auto-resume: if unchecked items remain, relaunch after cooldown
        local remaining
        remaining=$(grep -c '^\- \[ \]' "$TODO_FILE" 2>/dev/null || echo 0)
        if [ "$remaining" -gt 0 ]; then
            log "Still $remaining unchecked items. Waiting 60s before resuming..."
            sleep 60
            # Re-check lock in case something else launched
            if [ ! -f "$LOCK_FILE" ]; then
                launch_claude
            fi
        fi
    ) &
}

# Record initial state
PREV_COUNT=$(count_unchecked)
log "Watching $TODO_FILE ($PREV_COUNT unchecked items)"
log "Auto-resume enabled (60s cooldown). Step ordering enforced."
log "Press Ctrl+C to stop"

# If there are already unchecked items on startup, launch immediately
if [ "$PREV_COUNT" -gt 0 ] && ! check_and_clean_lock; then
    launch_claude
fi

# Watch for changes — also check lock health on every event
fswatch -o "$TODO_FILE" | while read -r _; do
    sleep 1  # debounce rapid saves

    # Always check lock health
    check_and_clean_lock || true

    NEW_COUNT=$(count_unchecked)
    if [ "$NEW_COUNT" -gt "$PREV_COUNT" ]; then
        log "New TODO items detected ($PREV_COUNT → $NEW_COUNT)"
        launch_claude
    elif [ "$NEW_COUNT" -lt "$PREV_COUNT" ]; then
        log "Items completed ($PREV_COUNT → $NEW_COUNT remaining)"
    fi
    PREV_COUNT=$NEW_COUNT
done
