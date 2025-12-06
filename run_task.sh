#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 POLICY_PATH ROBOT_PORT DURATION_SEC" >&2
  exit 1
fi

POLICY_PATH="$1"
ROBOT_PORT="$2"
DURATION="$3"

# Prefer GNU timeout if available. On macOS it may be gtimeout (brew install coreutils)
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
fi

# Dry-run mode can be forced via env DRY_RUN=1 (used on mac mini validation)
DRY_RUN="${DRY_RUN:-0}"
LOG_FILE="${LOG_FILE:-task_runs.log}"

if [ "$DRY_RUN" = "1" ] || ! command -v lerobot-control >/dev/null 2>&1; then
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  msg="[DRY-RUN ${ts}] lerobot-control --robot.type=so100_follower --robot.port=${ROBOT_PORT} --policy.path=${POLICY_PATH} --fps 30 (for ${DURATION}s)"
  echo "$msg" >&2
  # Append to log file for confirmation
  {
    echo "$msg"
  } >> "$LOG_FILE" 2>/dev/null || true
  sleep "${DURATION}"
  exit 0
fi

set -x
if [ -n "${TIMEOUT_BIN}" ]; then
  "${TIMEOUT_BIN}" "${DURATION}" lerobot-control \
    --robot.type=so100_follower \
    --robot.port="${ROBOT_PORT}" \
    --policy.path="${POLICY_PATH}" \
    --fps 30
else
  echo "[WARN] timeout command not available; running without enforced duration" >&2
  lerobot-control \
    --robot.type=so100_follower \
    --robot.port="${ROBOT_PORT}" \
    --policy.path="${POLICY_PATH}" \
    --fps 30
fi
