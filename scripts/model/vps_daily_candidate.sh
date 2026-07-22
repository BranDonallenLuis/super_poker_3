#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SUPER_POKER_PYTHON:-$ROOT/.venv/bin/python}"
DATA_DIR="${SUPER_POKER_DATA_DIR:-$ROOT/data/raw}"
ARTIFACTS_DIR="${SUPER_POKER_ARTIFACTS_DIR:-$ROOT/artifacts}"
LOG_DIR="${SUPER_POKER_LOG_DIR:-$ROOT/logs}"
LOCK_FILE="${SUPER_POKER_DAILY_LOCK:-/tmp/super-poker-3-daily-candidate.lock}"

mkdir -p "$DATA_DIR" "$ARTIFACTS_DIR/daily-history" "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[INFO] Daily candidate job is already running; skipping duplicate start."
    exit 0
fi

incumbent="$ARTIFACTS_DIR/super_poker_3.joblib"
before_hash=""
if [[ -f "$incumbent" ]]; then
    before_hash="$(sha256sum "$incumbent" | awk '{print $1}')"
fi

cd "$ROOT"
"$PYTHON_BIN" -m super_poker.automation daily \
    --data-dir "$DATA_DIR" \
    --artifacts "$ARTIFACTS_DIR" \
    --train-candidate

after_hash=""
if [[ -f "$incumbent" ]]; then
    after_hash="$(sha256sum "$incumbent" | awk '{print $1}')"
fi
if [[ "$before_hash" != "$after_hash" ]]; then
    echo "[ERROR] Daily candidate job changed the deployed artifact" >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
cp "$ARTIFACTS_DIR/automation-state.json" \
    "$ARTIFACTS_DIR/daily-history/$stamp.json"
echo "[INFO] Candidate evaluation recorded; deployed artifact remained unchanged."
