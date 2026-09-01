#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
BUCKET=${BUCKET:?set BUCKET to the durable S3 bucket}
RUN_ID=${RUN_ID:?set RUN_ID}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
PREFIX=${PREFIX:-runs/$RUN_ID}

mkdir -p "$ROOT/logs/checkpoint"
while true; do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if storecli sync "$ROOT" "deadstore:$BUCKET/$PREFIX" \
      --only-show-errors \
      --exclude 'cache/*' \
      --exclude 'source/*' \
      --exclude 'tmp/*'; then
    printf '%s ok\n' "$stamp" >>"$ROOT/logs/checkpoint/sync.log"
  else
    printf '%s failed\n' "$stamp" >>"$ROOT/logs/checkpoint/sync.log"
  fi
  sleep "$INTERVAL_SECONDS"
done
