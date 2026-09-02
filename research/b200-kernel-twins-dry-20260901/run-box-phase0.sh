#!/usr/bin/env bash
# First real-B200 window: fail-closed host/device preflight plus synthetic kernel correctness.
# --plan is host-only. `run` is the only mode that may open CUDA and requires explicit authority.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODE=${1:---plan}

usage() {
    cat <<'EOF'
usage: run-box-phase0.sh --plan | run

run requires:
  MEMRA_B200_HOST_ROLE=research-non-production
  MEMRA_B200_EXPECTED_SHA=<exact 40-hex commit>
  MEMRA_B200_RECEIPT_DIR=<new absolute directory>
optional:
  MEMRA_B200_DEVICE_LIST=0[,1...]   (default: 0)

Ordered run:
  host role + clean exact tree -> CUDA 13.1 -> all requested devices are CC 10.0 and idle
  -> /tmp/memra-gpu.lock -> dry gates -> release build -> per-device NVFP4 exact gate
  -> per-device FP8 exact/random/tail gate -> synthetic kernel-check -> sealed receipt manifest.
EOF
}

if [ "$MODE" = "--plan" ]; then
    usage
    exit 0
fi
[ "$MODE" = "run" ] || { usage >&2; exit 2; }

fail() { echo "b200-phase0: FAIL: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

[ "${MEMRA_B200_HOST_ROLE:-}" = "research-non-production" ] \
    || fail "MEMRA_B200_HOST_ROLE must be exactly research-non-production"
EXPECTED=${MEMRA_B200_EXPECTED_SHA:-}
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || fail "MEMRA_B200_EXPECTED_SHA must be 40 lowercase hex"
RECEIPT_DIR=${MEMRA_B200_RECEIPT_DIR:-}
[[ "$RECEIPT_DIR" = /* ]] || fail "MEMRA_B200_RECEIPT_DIR must be an absolute path"
DEVICES=${MEMRA_B200_DEVICE_LIST:-0}
[[ "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "MEMRA_B200_DEVICE_LIST must be comma-separated indices"

for cmd in git nvidia-smi flock cargo sha256sum tee awk grep pgrep find sort; do need "$cmd"; done
NVCC=${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}
[ -x "$NVCC" ] || fail "nvcc is not executable: $NVCC"

cd "$ROOT"
ACTUAL=$(git rev-parse HEAD)
[ "$ACTUAL" = "$EXPECTED" ] || fail "HEAD $ACTUAL != expected $EXPECTED"
[ -z "$(git status --porcelain)" ] || fail "worktree is dirty"

NVCC_VERSION=$("$NVCC" --version 2>&1)
grep -q 'release 13\.1' <<<"$NVCC_VERSION" || fail "CUDA toolkit is not 13.1"

GPU_CSV=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader,nounits) \
    || fail "nvidia-smi GPU query failed"
[ -n "$GPU_CSV" ] || fail "no visible GPU"
if ! awk -F, '
  { for (i=1; i<=NF; ++i) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
  $3 != "10.0" { exit 1 }
' <<<"$GPU_CSV"; then
    fail "every visible GPU must be compute capability 10.0"
fi
IFS=',' read -ra DEVICE_ARRAY <<<"$DEVICES"
for device in "${DEVICE_ARRAY[@]}"; do
    awk -F, -v want="$device" '
      { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); if ($1 == want) found=1 }
      END { exit !found }
    ' <<<"$GPU_CSV" || fail "requested device $device is not visible"
done

APPS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
[ -z "$APPS" ] || fail "GPU compute processes already exist: $APPS"
if pgrep -af '[m]emra-server|[r]un-gen|[r]un-spec|[k]ernel-check' >/dev/null; then
    fail "a Memra runtime/gate process is already running"
fi

exec 9>/tmp/memra-gpu.lock
flock -w 60 9 || { echo "b200-phase0: GPU lock timeout" >&2; exit 75; }

if [ -e "$RECEIPT_DIR" ] && [ -n "$(find "$RECEIPT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    fail "receipt directory already contains files: $RECEIPT_DIR"
fi
mkdir -p "$RECEIPT_DIR"

run_logged() {
    local label=$1
    shift
    local log="$RECEIPT_DIR/$label.log"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    [ "$rc" -eq 0 ] || fail "$label exited $rc (see $log)"
}

printf '%s\n' "$ACTUAL" > "$RECEIPT_DIR/git-head.txt"
printf '%s\n' "$GPU_CSV" > "$RECEIPT_DIR/gpus.csv"
printf '%s\n' "$NVCC_VERSION" > "$RECEIPT_DIR/nvcc.txt"
sha256sum \
  crates/memra-engine/build.rs \
  crates/memra-engine/cu/mmq_fp4.cu \
  crates/memra-engine/cu/mmq_nvfp4_w4a8.cu \
  crates/memra-engine/cu/mmq_fp8_blk.cu \
  crates/memra-engine/cu/sm100_blockscale_layout.cuh \
  crates/memra-engine/src/mmq_ffi.rs \
  crates/memra-engine/src/fp8_ffi.rs \
  crates/memra-engine/src/bin/nvfp4_mmq_check.rs \
  crates/memra-engine/src/bin/fp8_mmq_check.rs \
  research/b200-kernel-twins-dry-20260901/run-box-phase0.sh \
  > "$RECEIPT_DIR/source-manifest.sha256"

run_logged dry-layouts research/b200-kernel-twins-dry-20260901/check-layouts.sh
run_logged dry-nvfp4 research/b200-kernel-twins-dry-20260901/check-nvfp4.sh
run_logged dry-fp8 research/b200-kernel-twins-dry-20260901/check-fp8.sh
run_logged build-100a env MEMRA_NVCC="$NVCC" MEMRA_CUDA_ARCH=100a cargo build --release --bins

for device in "${DEVICE_ARRAY[@]}"; do
    run_logged "gpu${device}-nvfp4-exact" \
        env CUDA_VISIBLE_DEVICES="$device" MEMRA_FP8_MMQ=1 \
        target/release/nvfp4_mmq_check
    grep -q 'NVFP4-MMQ-EXACT ALL PASS' "$RECEIPT_DIR/gpu${device}-nvfp4-exact.log" \
        || fail "device $device NVFP4 log lacks ALL PASS"

    run_logged "gpu${device}-fp8-exact" \
        env CUDA_VISIBLE_DEVICES="$device" MEMRA_FP8_MMQ=1 \
        target/release/fp8_mmq_check
    grep -q 'fp8-mmq-check ALL GREEN' "$RECEIPT_DIR/gpu${device}-fp8-exact.log" \
        || fail "device $device FP8 log lacks ALL GREEN"

    run_logged "gpu${device}-kernel-check-fast" \
        env CUDA_VISIBLE_DEVICES="$device" MEMRA_FP8_MMQ=1 MEMRA_KC_FAST=1 \
        target/release/kernel-check
    grep -q '^ALL GREEN ' "$RECEIPT_DIR/gpu${device}-kernel-check-fast.log" \
        || fail "device $device kernel-check log lacks ALL GREEN"
done

printf 'PASS b200-phase0 sha=%s devices=%s\n' "$ACTUAL" "$DEVICES" \
    > "$RECEIPT_DIR/verdict.txt"
(
    cd "$RECEIPT_DIR"
    find . -maxdepth 1 -type f ! -name manifest.sha256 -printf '%P\n' \
        | sort | while read -r file; do sha256sum "$file"; done \
        > manifest.sha256
)
echo "b200-phase0: PASS receipts=$RECEIPT_DIR"
