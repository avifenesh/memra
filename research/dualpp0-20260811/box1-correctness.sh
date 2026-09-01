#!/usr/bin/env bash
# Dual-active PP-2 increment-0 model gates on box1, under one exclusive GPU hold.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp0}
OUT=${DUALPP_CORRECTNESS_OUT:-$REPO/research/dualpp0-20260811/raw/box1/correctness}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DUALPP_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
Q9=${DUALPP_Q9:-/home/ubuntu/smoke-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
PROMPT=${DUALPP_PROMPT:-$REPO/tools/fast-gate/prompts/probe.txt}
KERNEL=$REPO/target/release/kernel-check
BATCH=$REPO/target/release/decode-batch-gate
GEN=$REPO/target/release/run-gen
SPEC=$REPO/target/release/run-spec
MANIFEST=$REPO/tools/kernel-check-step35.cells
REFUSAL='decode_step_batch_dual: refused: PP boundary is single-slot; set MEMRA_PP_OVERLAP=1 so both alternating boundary slots are prepared before dual-active decode'
HOST_BOUNCE_REFUSAL='decode_step_batch_dual: refused: MEMRA_PP_HOST_BOUNCE=1 is unvalidated for dual-active decode; disable MEMRA_DUAL_PP or use peer transport'

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

for artifact in "$MODEL" "$DRAFT" "$Q9" "$PROMPT" "$KERNEL" "$BATCH" "$GEN" \
                "$SPEC" "$MANIFEST"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "CORRECTNESS_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git status --short --branch --untracked-files=no
sha256sum "$MODEL" "$DRAFT" "$Q9" "$PROMPT" "$KERNEL" "$BATCH" "$GEN" "$SPEC" \
    >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_logged kernel-check 7200 env -u MEMRA_PP_OVERLAP -u MEMRA_DUAL_PP \
    -u MEMRA_PP_HOST_BOUNCE \
    CUDA_VISIBLE_DEVICES=0,1 "$KERNEL" "$MODEL" --require-manifest "$MANIFEST"
grep -q 'dual-pp-wave-split .* OK' "$OUT/kernel-check.log"
grep -q 'dual-pp-single-slot-refusal .* OK' "$OUT/kernel-check.log"
grep -q 'dual-pp-hostbounce-refusal .* OK' "$OUT/kernel-check.log"
grep -q 'ALL GREEN (' "$OUT/kernel-check.log"

# The gate itself first runs dual-ON + single-slot and requires the quoted refusal with no
# cache advance. It then opens the double slot and compares every logit bit at c=1..8 while
# requiring DUAL_PP_OVERLAPS to advance for every c>=2 replay. Serving c=16 is scheduler-
# chunked into this checkpoint's exact B<=8 tier; the perf harness proves that live shape.
run_logged dual-matrix 10800 env -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_DUAL_PP=1 MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 "$BATCH" "$MODEL" --mode pp \
    --batch 1,2,3,4,5,6,7,8 --steps 8 --reps 1 --stages 2 --plen 520
grep -Fq "dual pp negative PASS: $REFUSAL" "$OUT/dual-matrix.log"
grep -Fq "dual pp host-bounce negative PASS: $HOST_BOUNCE_REFUSAL" "$OUT/dual-matrix.log"
test "$(grep -c 'dual pp liveness PASS' "$OUT/dual-matrix.log")" -eq 7
grep -q 'pp mode verdict: 0 failing arm(s)' "$OUT/dual-matrix.log"
grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' "$OUT/dual-matrix.log"

run_logged strict-batch 3600 env CUDA_VISIBLE_DEVICES=0 MEMRA_MMVQ=0 \
    MEMRA_NO_FUSE_NORMQ=1 "$BATCH" "$Q9" --steps 32 --batch 4 --mode strict
grep -q 'ALL GREEN: decode_step_batch exactness battery' "$OUT/strict-batch.log"

run_logged run-gen 3600 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT" "$GEN" "$MODEL"
test "$(grep -c 'MATCH' "$OUT/run-gen.log")" -ge 2

run_logged run-spec 7200 env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_NGEN=32 MEMRA_MTP_DRAFT="$DRAFT" MEMRA_PROMPT_FILE="$PROMPT" \
    "$SPEC" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "CORRECTNESS_PASS $(date -u +%FT%TZ)"
