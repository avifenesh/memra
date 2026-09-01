#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
BUCKET=${BUCKET:?set BUCKET to the durable S3 bucket}
RUN_ID=${RUN_ID:?set RUN_ID}
PREFIX=${PREFIX:-runs/$RUN_ID}
IMDS=http://169.254.169.254/latest
IMDS_TOKEN_HEADER="<provider-metadata-token-header>"

mkdir -p "$ROOT/logs/interruption"
while true; do
  token=$(curl -fsS -X PUT "$IMDS/api/token" -H "$IMDS_TOKEN_HEADER-ttl-seconds: 60")
  if notice=$(curl -fsS -H "$IMDS_TOKEN_HEADER: $token" \
      "$IMDS/<spot-interruption-endpoint>" 2>/dev/null); then
    python3 - "$ROOT/logs/interruption/notice.json" "$notice" <<'PY'
import json
import pathlib
import sys
payload = json.loads(sys.argv[2])
payload["observed_by"] = "bw24-hy3-110gb-spot-watch-v1"
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    timeout 90 storecli sync "$ROOT" "deadstore:$BUCKET/$PREFIX" \
      --only-show-errors \
      --exclude 'cache/*' \
      --exclude 'source/*' \
      --exclude 'tmp/*' || true
    storecli cp "$ROOT/logs/interruption/notice.json" \
      "deadstore:$BUCKET/$PREFIX/INTERRUPTION-NOTICE.json" --only-show-errors || true
    exit 0
  fi
  sleep 5
done
