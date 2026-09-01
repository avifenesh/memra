#!/usr/bin/env bash
# Pinned-host transfer screening for the cache-tier design. This is not an end-to-end
# checkpoint benchmark: CUDA Samples explicitly warns that bandwidthTest is not a performance
# tool. The result only bounds whether a host tier is worth prototyping against measured prefill.
set -uo pipefail

REPO=${REPO:-$HOME/memra-cx-cachespec}
TS=${CACHESPEC_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CACHESPEC_OUT:-$REPO/research/cachespec-20260809/raw/box1/h2d-screen-$TS.log}
BANDWIDTH_TEST=${BANDWIDTH_TEST:-/usr/local/cuda-12.9/extras/demo_suite/bandwidthTest}
START_BYTES=${START_BYTES:-536870912}
END_BYTES=${END_BYTES:-1073741824}
INCREMENT_BYTES=${INCREMENT_BYTES:-536870912}

mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

exec 9>/tmp/memra-gpu.lock
flock -w "${LOCK_WAIT:-14400}" 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
trap 'flock -u 9; exec 9>&-' EXIT

echo "HOST_TIER_SCREEN_BEGIN $(date -u +%FT%TZ)"
echo "host=$(hostname) tool=$BANDWIDTH_TEST"
echo "sizes=$START_BYTES..$END_BYTES step=$INCREMENT_BYTES memory=pinned"
echo "CAUTION: bandwidthTest is a transfer screening tool, not an end-to-end cache restore benchmark."
test -x "$BANDWIDTH_TEST" || {
    echo "FAIL: missing executable $BANDWIDTH_TEST"
    exit 1
}

grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo
nvidia-smi --query-gpu=index,name,temperature.gpu,pstate,clocks.sm,power.draw,memory.used,memory.free \
    --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader || true

for device in 0 1; do
    echo "DEVICE=$device DIRECTION=H2D"
    "$BANDWIDTH_TEST" --csv --device="$device" --memory=pinned --mode=range \
        --start="$START_BYTES" --end="$END_BYTES" --increment="$INCREMENT_BYTES" --htod
    echo "DEVICE=$device DIRECTION=D2H"
    "$BANDWIDTH_TEST" --csv --device="$device" --memory=pinned --mode=range \
        --start="$START_BYTES" --end="$END_BYTES" --increment="$INCREMENT_BYTES" --dtoh
done

nvidia-smi --query-gpu=index,name,temperature.gpu,pstate,clocks.sm,power.draw,memory.used,memory.free \
    --format=csv,noheader
echo "HOST_TIER_SCREEN_DONE $(date -u +%FT%TZ)"

flock -u 9
exec 9>&-
trap - EXIT
