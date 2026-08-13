#!/bin/bash
# Runs a script inside a detached tmux session, so it survives an SSH/VS Code
# Remote disconnect -- the session keeps running on the remote machine
# regardless of whether your client is still connected to it.
#
# Usage:
#   ./run_in_tmux.sh scripts/tests/test_dynamic_global_tilts_effect.py
#   ./run_in_tmux.sh scripts/tests/test_dynamic_global_tilts_effect.py --some-arg   # extra args pass through
#
# Then:
#   tmux attach -t <session_name>   # reattach and watch it live
#   Ctrl+B, D                       # detach again (leaves it running)
#   tmux kill-session -t <session_name>   # stop it early
#
# The session name is the script's own basename (e.g. "test_dynamic_global_tilts_effect"),
# so re-running the same script while a session for it is still up refuses rather
# than silently starting a second, competing run -- kill the old one first if you
# want to restart from scratch.
#
# Runs Python with -u (unbuffered stdout): piping through tee otherwise makes
# Python fully block-buffer its output instead of line-buffering it, so the log
# (and the tmux pane) can sit empty for a long time even though the process is
# genuinely running -- print() output only shows up once the internal buffer
# fills or the process exits.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/shadab/venvs/sionna-wbbf/bin/python3"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script_path> [args...]" >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed (or not on PATH)." >&2
    exit 1
fi

SCRIPT_PATH="$1"
shift
SESSION_NAME="$(basename "$SCRIPT_PATH" .py)"
LOG_DIR="$REPO_ROOT/results/tests/_run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "A tmux session named '$SESSION_NAME' is already running." >&2
    echo "Attach:      tmux attach -t $SESSION_NAME" >&2
    echo "Or stop it:  tmux kill-session -t $SESSION_NAME" >&2
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
    "cd '$REPO_ROOT' && '$VENV_PYTHON' -u '$SCRIPT_PATH' $* 2>&1 | tee '$LOG_FILE'; echo; echo '[done -- press any key to close]'; read -n 1"

echo "Started: $SCRIPT_PATH"
echo "Session: $SESSION_NAME"
echo "Log:     $LOG_FILE"
echo
echo "Attach:  tmux attach -t $SESSION_NAME"
echo "Detach again: Ctrl+B, then D (leaves it running)"
