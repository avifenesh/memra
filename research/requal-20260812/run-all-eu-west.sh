#!/usr/bin/env bash
# Hold the shared GPU flock continuously across the fixed sellgate gates and campaign.
set -euo pipefail

ROOT=${REQUAL_ROOT:-/opt/dl-image/nvme/cx-requal}
MODELS=$ROOT/models
DRIVER=$ROOT/harness/run-eu-west.sh
STAMP=${REQUAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${REQUAL_PIPELINE_OUT:-$ROOT/raw/pipeline-$STAMP}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/pipeline.log") 2>&1

echo "REQUAL_PIPELINE_START ts=$(date -u +%FT%TZ) pid=$$"
echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "REQUAL_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"

export SELLGATE_LOCK_HELD=1
export SELLGATE_ROOT=$ROOT
export SELLGATE_MODELS=$MODELS
export SELLGATE_STAMP=$STAMP
"$DRIVER" gates
touch "$OUT/gates-complete.ok"
"$DRIVER" campaign
touch "$OUT/campaign-complete.ok"

echo "REQUAL_PIPELINE_PASS ts=$(date -u +%FT%TZ)"
