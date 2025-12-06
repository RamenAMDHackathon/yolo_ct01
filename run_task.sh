#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 POLICY_PATH ROBOT_PORT DURATION_SEC" >&2
  exit 1
fi

POLICY_PATH="$1"
ROBOT_PORT="$2"
DURATION="$3"

# Allow overriding control binary and robot type via env
# CONTROL_BIN=lerobot-control|lerobot-record (default: lerobot-record)
# ROBOT_TYPE defaults to so101_follower to match user's setup
CONTROL_BIN="${CONTROL_BIN:-lerobot-record}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
# Fixed arguments embedded to avoid shell word-splitting issues
# These replace complex values that previously came via EXTRA_ARGS
FPS="30"
ROBOT_ID="my_awesome_follower_arm"
ROBOT_CAMERAS='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}'
DATASET_SINGLE_TASK='put the pen in the white square'
DATASET_REPO_ID='AmdRamen/eval_mission_1'
DATASET_ROOT='/home/amddemo/hackathon_ramen/outputs/eval_lerobot_dataset/'
DATASET_EPISODE_TIME_S='20'
DATASET_NUM_EPISODES='5'
DATASET_PUSH_TO_HUB='false'

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

if [ "$DRY_RUN" = "1" ] || ! command -v "$CONTROL_BIN" >/dev/null 2>&1; then
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  msg="[DRY-RUN ${ts}] ${CONTROL_BIN} --robot.type=${ROBOT_TYPE} --robot.port=${ROBOT_PORT} --policy.path=${POLICY_PATH} --fps ${FPS} --robot.id=${ROBOT_ID} --robot.cameras=\"${ROBOT_CAMERAS}\" --dataset.single_task=\"${DATASET_SINGLE_TASK}\" --dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT} --dataset.episode_time_s=${DATASET_EPISODE_TIME_S} --dataset.num_episodes=${DATASET_NUM_EPISODES} --dataset.push_to_hub=${DATASET_PUSH_TO_HUB} (for ${DURATION}s)"
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
  "${TIMEOUT_BIN}" "${DURATION}" "${CONTROL_BIN}" \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --policy.path="${POLICY_PATH}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${ROBOT_CAMERAS}" \
    --dataset.single_task="${DATASET_SINGLE_TASK}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episode_time_s="${DATASET_EPISODE_TIME_S}" \
    --dataset.num_episodes="${DATASET_NUM_EPISODES}" \
    --dataset.push_to_hub="${DATASET_PUSH_TO_HUB}"
else
  echo "[WARN] timeout command not available; running without enforced duration" >&2
  "${CONTROL_BIN}" \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --policy.path="${POLICY_PATH}" \や manager.py のようなもの）の中で、以下のように cap.release() を挟んでください。
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${ROBOT_CAMERAS}" \
    --dataset.single_task="${DATASET_SINGLE_TASK}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episode_time_s="${DATASET_EPISODE_TIME_S}" \
    --dataset.num_episodes="${DATASET_NUM_EPISODES}" \
    --dataset.push_to_hub="${DATASET_PUSH_TO_HUB}"
fi
