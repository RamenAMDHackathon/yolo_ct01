#!/usr/bin/env sh
# POSIX sh 互換。dash 等でも動くように pipefail は使いません。
set -eu

# Helper to apply initial pose. In this repo it only logs the intent.
# If you have a real control tool, replace the echo section with actual commands.

ROBOT_TYPE="${1:-${ROBOT_TYPE:-so101_follower}}"
ROBOT_PORT="${2:-${ROBOT_PORT:-/dev/ttyACM1}}"
POSE_ENV="${ROBOT_INIT_POSE:-}"

echo "[apply_initial_pose] robot.type=${ROBOT_TYPE} robot.port=${ROBOT_PORT}"
if [ -n "$POSE_ENV" ]; then
  echo "[apply_initial_pose] pose(norm): $POSE_ENV"
else
  echo "[apply_initial_pose] pose(norm): (not provided via env)"
fi

# Example: Here is where you could call your actual control binary
# CONTROL_BIN could be something like 'lerobot-control' if it supports setting joints.
# This is left as a no-op to avoid errors in validation environments.

exit 0
