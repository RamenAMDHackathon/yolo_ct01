#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"
PY="$VENV/bin/python"

# Override via environment variables if needed
CAMERA_ID="${CAMERA_ID:-1}"
CONF="${CONF:-0.5}"
WEIGHTS="${WEIGHTS:-yolov8n.pt}"
DEVICE="${DEVICE:-auto}"  # auto|cpu|mps|cuda|cuda:0
OAK_WIDTH="${OAK_WIDTH:-640}"
OAK_HEIGHT="${OAK_HEIGHT:-480}"
DEPTH_KSIZE="${DEPTH_KSIZE:-5}"

exec "$PY" garbage_sort_yolo.py \
  --use-oakd -c "$CAMERA_ID" -t "$CONF" -w "$WEIGHTS" -d "$DEVICE" \
  --oak-width "$OAK_WIDTH" --oak-height "$OAK_HEIGHT" --depth-ksize "$DEPTH_KSIZE"

