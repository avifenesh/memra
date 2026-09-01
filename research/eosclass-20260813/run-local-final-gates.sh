#!/usr/bin/env bash
# Local 5090 correctness battery for the one-program serving default.
set -euo pipefail

cd "$(dirname "$0")/../.."

LABEL=${1:?usage: $0 LABEL}
OUT=research/eosclass-20260813/raw/$LABEL
LOCK=/tmp/memra-5090.lock
LOCK_WAIT=${EOSCLASS_LOCK_WAIT_SECONDS:-7200}
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
PROMPT=research/e2e/prompts/pp512.txt
KERNEL=target/release/kernel-check
BATCH_GATE=target/release/decode-batch-gate
RUN_GEN=target/release/run-gen
RUN_SPEC=target/release/run-spec
SERVER=target/release/memra-server

for path in "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$PROMPT" \
    "$KERNEL" "$BATCH_GATE" "$RUN_GEN" "$RUN_SPEC" "$SERVER"; do
    test -e "$path" || { echo "missing required path: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }

exec 9>"$LOCK"
echo "waiting up to ${LOCK_WAIT}s for GPU lease: $LOCK" >&2
if ! flock -w "$LOCK_WAIT" 9; then
    echo "GPU lease busy or wait expired: $LOCK" >&2
    exit 75
fi
mkdir -p "$OUT"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "head=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    git status --short
    sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$PROMPT" \
        "$KERNEL" "$BATCH_GATE" "$RUN_GEN" "$RUN_SPEC" "$SERVER" \
        tools/serve-smoke.sh tools/q35-cold-mixed-gate.py
    echo "gpu_lock=$LOCK"
    echo "gpu_lock_wait_seconds=$LOCK_WAIT"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
} >"$OUT/provenance.log" 2>&1

run_logged() {
    local label=$1
    shift
    set +e
    timeout 7200 "$@" 2>&1 | tee "$OUT/$label.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/$label.exit"
    return "$rc"
}

failed=0

for model_name in q27 q35; do
    if test "$model_name" = q27; then model=$Q27; else model=$Q35; fi
    run_logged "kernel-check-$model_name" env \
        -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS -u MEMRA_MOE_GROUPED \
        CUDA_VISIBLE_DEVICES=0 "$KERNEL" "$model" || failed=1
    if ! grep -q 'ALL GREEN' "$OUT/kernel-check-$model_name.log" \
        || grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' \
            "$OUT/kernel-check-$model_name.log"; then
        failed=1
    fi
done

for model_name in q27 q35; do
    if test "$model_name" = q27; then model=$Q27; else model=$Q35; fi
    run_logged "decode-batch-$model_name-config" env \
        -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
        CUDA_VISIBLE_DEVICES=0 "$BATCH_GATE" "$model" \
        --steps 32 --batch 4 --mode config || failed=1
    if ! grep -q 'ALL GREEN' "$OUT/decode-batch-$model_name-config.log" \
        || ! grep -Eq 'global setting = OFF; effective .* = OFF' \
            "$OUT/decode-batch-$model_name-config.log"; then
        failed=1
    fi
done

run_logged decode-batch-q27-strict env \
    -u MEMRA_SERVE_GS \
    CUDA_VISIBLE_DEVICES=0 MEMRA_SERVE_B1FAST=1 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    "$BATCH_GATE" "$Q27" --steps 32 --batch 4 --mode strict || failed=1
grep -q 'ALL GREEN' "$OUT/decode-batch-q27-strict.log" || failed=1

for model_name in q27 q35; do
    if test "$model_name" = q27; then model=$Q27; else model=$Q35; fi
    run_logged "run-gen-$model_name" env \
        -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS -u MEMRA_MOE_GROUPED \
        CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
        "$RUN_GEN" "$model" || failed=1
    if ! grep -q 'argmax=.*MATCH' "$OUT/run-gen-$model_name.log" \
        || grep -q 'MISMATCH' "$OUT/run-gen-$model_name.log"; then
        failed=1
    fi
done

for model_name in q27 q35; do
    if test "$model_name" = q27; then
        model=$Q27
        draft=$Q27_DRAFT
    else
        model=$Q35
        draft=$Q35_DRAFT
    fi
    run_logged "run-spec-$model_name" env \
        -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_SERVE_B1FAST \
        -u MEMRA_SERVE_GS -u MEMRA_MOE_GROUPED \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$draft" MEMRA_NGEN=32 \
        MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$model" || failed=1
    if test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-$model_name.log")" -ne 8 \
        || ! grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-$model_name.log" \
        || grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-$model_name.log"; then
        failed=1
    fi
done

run_logged serve-smoke env \
    -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS -u MEMRA_MOE_GROUPED \
    TMPDIR=/home/avifenesh/tmp-lanes CARGO_BUILD_JOBS=1 CUDA_VISIBLE_DEVICES=0 \
    MEMRA_Q35_COLD_MODEL="$Q35" tools/serve-smoke.sh || failed=1
grep -q 'serve-smoke: 0 failed' "$OUT/serve-smoke.log" || failed=1
test -f /tmp/serve-smoke-q35-cold-mixed.log \
    && cp /tmp/serve-smoke-q35-cold-mixed.log "$OUT/q35-cold-mixed.jsonl"
test -f /tmp/serve-smoke.log && cp /tmp/serve-smoke.log "$OUT/serve-smoke-server-last.log"

nvidia-smi --query-gpu=index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$OUT/gpu-after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/compute-apps-after.csv" || true
grep -Ein 'out of memory|CUDA_ERROR|panic|fatal|illegal address|misaligned address' \
    "$OUT"/*.log >"$OUT/failure-signature-scan.log" || true
echo "$failed" >"$OUT/overall.exit"
exit "$failed"
