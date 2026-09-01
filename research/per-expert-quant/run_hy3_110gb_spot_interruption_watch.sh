#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
BUCKET=${BUCKET:?set BUCKET to the durable S3 bucket}
RUN_ID=${RUN_ID:?set RUN_ID}
PREFIX=${PREFIX:-runs/$RUN_ID}
IMDS=http://169.254.169.254/latest
IMDS_TOKEN_HEADER=$(printf 'X-hyperscaler-e%s2-metadata-token' c)

mkdir -p "$ROOT/logs/interruption"
while true; do
  token=$(curl -fsS -X PUT "$IMDS/api/token" -H "$IMDS_TOKEN_HEADER-ttl-seconds: 60")
  if notice=$(curl -fsS -H "$IMDS_TOKEN_HEADER: $token" \
      "$IMDS/meta-data/spot/instance-action" 2>/dev/null); then
    python3 - "$ROOT/logs/interruption/notice.json" "$notice" <<'PY'
import json
import pathlib
import sys
payload = json.loads(sys.argv[2])
payload["observed_by"] = "bw24-hy3-110gb-spot-watch-v1"
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    timeout 90 hyperscaler s3 sync "$ROOT" "obj://$BUCKET/$PREFIX" \
      --only-show-errors \
      --exclude 'cache/*' \
      --exclude 'source/*' \
      --exclude 'tmp/*' || true
    hyperscaler s3 cp "$ROOT/logs/interruption/notice.json" \
      "obj://$BUCKET/$PREFIX/INTERRUPTION-NOTICE.json" --only-show-errors || true
    exit 0
  fi
  sleep 5
done
