#!/usr/bin/env bash
# Queue atomically behind the current local owner, build once, then reproduce the four committed
# split cells with targeted field detail. No GPU work runs before the shared lock is acquired.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${SPLITISO_SMOKE_OUT:-$REPO/research/splitiso-20260813/raw/local-smoke}
CELL=$OUT/original-four
LOCK=/tmp/memra-5090.lock

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/orchestrator.log") 2>&1

echo "LOCAL_SMOKE_QUEUE ts=$(date -u +%FT%TZ) lock=$LOCK source=$(git -C "$REPO" rev-parse HEAD)"
exec 9>"$LOCK"
flock -w 14400 9 || { echo "FAIL: local GPU lock wait timed out"; exit 75; }
echo "LOCAL_SMOKE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ)"

apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
if test -n "$apps"; then
    printf '%s\n' "$apps"
    echo "FAIL: local GPU has a compute process after lock acquisition"
    exit 1
fi

set +e
TMPDIR=/home/avifenesh/tmp-lanes cargo build --manifest-path "$REPO/Cargo.toml" \
    --release -p memra-server 2>&1 | tee "$OUT/release-build.log"
build_rc=${PIPESTATUS[0]}
set -e
test "$build_rc" -eq 0
sha256sum "$REPO/target/release/memra-server" | tee "$OUT/server-sha256.log"

SPLITISO_EXPECTED_SOURCE=$(git -C "$REPO" rev-parse HEAD) \
SPLITISO_OUT="$CELL" \
SPLITISO_SPLITS=64,512,2048,4374 \
SPLITISO_DETAIL_BOUNDARIES=64,512,2048,4374 \
SPLITISO_GPU_LOCK_HELD=1 \
    "$REPO/research/splitiso-20260813/run-local-cell.sh"

test -z "$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
echo "LOCAL_SMOKE_COMPLETE ts=$(date -u +%FT%TZ)"
