#!/usr/bin/env bash
# Final-source PP-2 speculative-decode correctness gates on box1.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"

REPO=${SPEC_PP2FIX_REPO:-/home/ubuntu/memra-cx-specpp2fix}
OUT=${SPEC_PP2FIX_GATES_OUT:-$REPO/research/specpp2fix-20260812/raw/box1/gates}
MODEL_ROOT=${SPEC_PP2FIX_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SPEC_PP2FIX_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${SPEC_PP2FIX_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PROMPT=${SPEC_PP2FIX_PROMPT:-$REPO/tools/fast-gate/prompts/probe.txt}
KERNEL=$REPO/target/release/kernel-check
BATCH=$REPO/target/release/decode-batch-gate
GEN=$REPO/target/release/run-gen
SPEC=$REPO/target/release/run-spec
MANIFEST=$REPO/tools/kernel-check-step35.cells

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

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

run_logged() {
    local label=$1 timeout_s=$2
    shift 2
    echo "gate_start=$label ts=$(date -u +%FT%TZ)"
    set +e
    timeout "$timeout_s" "$@" 2>&1 | tee "$OUT/$label.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/$label.exit"
    echo "gate_done=$label ts=$(date -u +%FT%TZ) rc=$rc"
    return "$rc"
}

for artifact in "$MODEL" "$DRAFT" "$PROMPT" "$KERNEL" "$BATCH" "$GEN" "$SPEC" \
                "$MANIFEST"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
sha256sum "$MODEL" "$DRAFT" "$PROMPT" "$KERNEL" "$BATCH" "$GEN" "$SPEC" \
    >"$OUT/SHA256SUMS"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GATES_LOCK_ACQUIRED $(date -u +%FT%TZ)"
snapshot "$OUT/nvidia-smi-before.log" gates-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_logged kernel-check 7200 env -u MEMRA_PP_OVERLAP -u MEMRA_DUAL_PP \
    -u MEMRA_PP_HOST_BOUNCE CUDA_VISIBLE_DEVICES=0,1 \
    "$KERNEL" "$MODEL" --require-manifest "$MANIFEST"
grep -q 'ALL GREEN (' "$OUT/kernel-check.log"

for placement in 0,1 1,0; do
    suffix=${placement/,/}
    run_logged "ppspec-dev$suffix" 7200 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES="$placement" \
        MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$BATCH" "$MODEL" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
    grep -q 'ppspec mode verdict: 0 failing arm(s)' "$OUT/ppspec-dev$suffix.log"
    grep -q 'ALL GREEN: spec-verify PP-2 stage-split exactness battery' \
        "$OUT/ppspec-dev$suffix.log"
done

run_logged ppbatch-dev01 7200 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    "$BATCH" "$MODEL" --mode pp --stages 2 --steps 16 --batch 1,4,8 --reps 2
grep -q 'pp mode verdict: 0 failing arm(s)' "$OUT/ppbatch-dev01.log"
grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' "$OUT/ppbatch-dev01.log"

run_logged run-gen 7200 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT" "$GEN" "$MODEL"
test "$(grep -c 'MATCH' "$OUT/run-gen.log")" -ge 2
if grep -q 'MISMATCH' "$OUT/run-gen.log"; then
    echo "FAIL: run-gen emitted MISMATCH"
    exit 1
fi

for placement in 0,1 1,0; do
    suffix=${placement/,/}
    run_logged "run-spec-dev$suffix" 10800 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
        -u MEMRA_SPEC_K CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES="$placement" MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 MEMRA_NGEN=32 MEMRA_MTP_DRAFT="$DRAFT" \
        MEMRA_PROMPT_FILE="$PROMPT" "$SPEC" "$MODEL"
    test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-dev$suffix.log")" -eq 8
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-dev$suffix.log"
done

grep -En \
    'CUDA_ERROR|illegal memory access|ILLEGAL_ADDRESS|sentinel|panicked at|MISMATCH|self-consistency: FAIL' \
    "$OUT"/kernel-check.log "$OUT"/pp*.log "$OUT"/run-*.log \
    >"$OUT/failure-scan.log" || true
test ! -s "$OUT/failure-scan.log" || { cat "$OUT/failure-scan.log"; exit 1; }

snapshot "$OUT/nvidia-smi-after.log" gates-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "GATES_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
echo "SPEC_PP2FIX_GATES_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
