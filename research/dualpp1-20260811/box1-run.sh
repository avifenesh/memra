#!/usr/bin/env bash
# Detached increment-1 box1 driver: one lock hold from release rebuild through all evidence.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp1}
ROOT=${DUALPP_BOX1_OUT:-$REPO/research/dualpp1-20260811/raw/box1}
CARGO=${CARGO:-/home/ubuntu/.cargo/bin/cargo}

test ! -e "$ROOT" || { echo "FAIL: output already exists: $ROOT"; exit 1; }
mkdir -p "$ROOT/build"
exec > >(tee "$ROOT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
export DUALPP_LOCK_HELD=1
echo "DUALPP1_LOCK_ACQUIRED $(date -u +%FT%TZ) pid=$$ ppid=$PPID sid=$(ps -o sid= -p $$ | tr -d ' ')"

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git status --short --branch --untracked-files=no
git diff --quiet
git diff --cached --quiet
test -x "$CARGO"
snapshot "$ROOT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

echo "BUILD_START $(date -u +%FT%TZ)"
set +e
/usr/bin/time -f 'wall_s=%e max_rss_kb=%M exit=%x' \
    "$CARGO" build --release -p memra-server 2>&1 | tee "$ROOT/build/server.log"
server_rc=${PIPESTATUS[0]}
set -e
echo "$server_rc" >"$ROOT/build/server.exit"
test "$server_rc" -eq 0

set +e
/usr/bin/time -f 'wall_s=%e max_rss_kb=%M exit=%x' \
    "$CARGO" build --release -p memra-engine \
    --bin kernel-check --bin decode-batch-gate --bin run-gen --bin run-spec \
    2>&1 | tee "$ROOT/build/gates.log"
gates_rc=${PIPESTATUS[0]}
set -e
echo "$gates_rc" >"$ROOT/build/gates.exit"
test "$gates_rc" -eq 0
echo "BUILD_PASS $(date -u +%FT%TZ)"

sha256sum target/release/memra-server target/release/kernel-check \
    target/release/decode-batch-gate target/release/run-gen target/release/run-spec \
    >"$ROOT/build/SHA256SUMS"
snapshot "$ROOT/nvidia-smi-post-build.log" release-built

EXPECTED_SOURCE="$EXPECTED_SOURCE" DUALPP_REPO="$REPO" \
    DUALPP_CORRECTNESS_OUT="$ROOT/correctness" \
    research/dualpp1-20260811/box1-correctness.sh

EXPECTED_SOURCE="$EXPECTED_SOURCE" DUALPP_REPO="$REPO" \
    DUALPP_SOAK_OUT="$ROOT/soak" \
    research/dualpp1-20260811/box1-soak.sh

EXPECTED_SOURCE="$EXPECTED_SOURCE" DUALPP_REPO="$REPO" \
    DUALPP_PERF_OUT="$ROOT/perf" \
    research/dualpp1-20260811/box1-perf.sh

snapshot "$ROOT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "DUALPP1_ALL_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
