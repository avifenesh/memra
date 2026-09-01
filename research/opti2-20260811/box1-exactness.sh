#!/usr/bin/env bash
# One-lock box1 exactness battery for OPTIPIPE increment 2 (serial 10/10 is separate).
set -euo pipefail

ROOT=${OPTI2_ROOT:-/home/ubuntu/memra-opti2}
OUT=${OPTI2_EXACT_OUT:-/home/ubuntu/opti2-receipts/exact-gates-1}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
BIN=${ROOT}/target/release

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$BIN/run-spec" "$BIN/run-gen" \
        "$BIN/kernel-check" "$BIN/optipipe-gate"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

snapshot() {
    local path=$1
    {
        date -u +%FT%TZ
        nvidia-smi \
            --query-gpu=index,name,memory.used,memory.total,temperature.gpu,pstate,clocks.sm,power.draw \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } > "$path" 2>&1
}

run_capture() {
    local name=$1
    shift
    echo "=== $name $(date -u +%FT%TZ) ==="
    "$@" > >(tee "$OUT/${name}.log") 2>&1
    echo "PASS: $name rc=0 $(date -u +%FT%TZ)"
}

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "EXACT_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch
sha256sum "$MODEL" "$DRAFT" "$BIN/run-spec" "$BIN/run-gen" \
    "$BIN/kernel-check" "$BIN/optipipe-gate" > "$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || {
    echo "FAIL: box1 not GPU-idle"
    exit 1
}

run_capture kernel-check timeout 3600 env CUDA_VISIBLE_DEVICES=0 \
    "$BIN/kernel-check" "$MODEL"
grep -q 'ALL GREEN: kernels match CPU reference' "$OUT/kernel-check.log"
test "$(grep -c ' OK' "$OUT/kernel-check.log")" -gt 300

run_capture run-gen timeout 3600 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_NGEN=64 \
    "$BIN/run-gen" "$MODEL" --prompt \
    'Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard.'
test "$(grep -c ' MATCH' "$OUT/run-gen.log")" -eq 2
! grep -q 'MISMATCH' "$OUT/run-gen.log"

run_capture run-spec timeout 3600 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_MTP_DRAFT="$DRAFT" \
    MEMRA_NGEN=32 \
    "$BIN/run-spec" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"

for threshold in 0.0 0.5 0.7 0.9; do
    tag=${threshold/./}
    run_capture "controller-q${tag}" timeout 3600 env \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_MTP_DRAFT="$DRAFT" \
        MEMRA_SPEC_GATE=0 \
        MEMRA_SPEC_K=1 \
        MEMRA_SPEC_DEVACC=1 \
        MEMRA_NGEN=64 \
        MEMRA_OPTI_CONTROLLER_Q="$threshold" \
        "$BIN/optipipe-gate" "$MODEL" controller
    grep -q 'STATE IDENTITY: PASS mode=controller' "$OUT/controller-q${tag}.log"
    grep -q 'RECURRENT RESTORE PRIMITIVE: PASS' "$OUT/controller-q${tag}.log"
done

snapshot "$OUT/nvidia-smi-after.log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
echo "EXACT_PASS $(date -u +%FT%TZ)"
