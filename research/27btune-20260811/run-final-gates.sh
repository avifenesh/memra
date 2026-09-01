#!/usr/bin/env bash
# Final local RTX 5090 correctness battery for the promoted naked default.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
RAW=$ROOT/research/27btune-20260811/raw
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
PROMPT=$ROOT/research/e2e/prompts/p1-code-short.txt

test -x "$ROOT/target/release/kernel-check"
test -x "$ROOT/target/release/run-gen"
test -x "$ROOT/target/release/run-spec"
test -f "$MODEL"
test -f "$DRAFT"

echo "lock-request $(date -u +%FT%TZ) wait=1200s"
exec 9>/tmp/memra-gpu.lock
flock -w 1200 9 || {
    echo "GPU lock unavailable after 1200s" >&2
    exit 75
}

gpu_state() {
    echo "[gpu $(date -u +%FT%TZ)]"
    nvidia-smi \
        --query-gpu=index,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        | sed 's/^/[compute-app] /'
}

run_logged() {
    local label=$1
    shift
    echo "=== gate=$label start=$(date -u +%FT%TZ) ==="
    gpu_state
    set +e
    "$@" 2>&1 | tee "$RAW/final-$label.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "=== gate=$label end=$(date -u +%FT%TZ) rc=$rc ==="
    return "$rc"
}

echo "capture-start $(date -u +%FT%TZ)"
echo "tree $(git -C "$ROOT" rev-parse HEAD)"
echo "model-sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
echo "draft-sha256 $(sha256sum "$DRAFT" | cut -d' ' -f1)"
echo "prompt-sha256 $(sha256sum "$PROMPT" | cut -d' ' -f1)"
sha256sum "$ROOT/target/release/kernel-check" "$ROOT/target/release/run-gen" \
    "$ROOT/target/release/run-spec"
gpu_state

run_logged kernel-check timeout 3600 "$ROOT/target/release/kernel-check" "$MODEL" \
    --require-manifest "$ROOT/tools/kernel-check-27b.cells"
grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' \
    "$RAW/final-kernel-check.log"

run_logged run-gen timeout 1800 env \
    -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_PROFILE_SPEC \
    CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$ROOT/target/release/run-gen" "$MODEL"
grep -qE 'argmax=.* MATCH$' "$RAW/final-run-gen.log"
if grep -q 'MISMATCH' "$RAW/final-run-gen.log"; then
    echo "run-gen reported MISMATCH" >&2
    exit 1
fi

run_logged run-spec timeout 3600 env \
    -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_SPEC_K -u MEMRA_PROFILE_SPEC \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=64 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$ROOT/target/release/run-spec" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$RAW/final-run-spec.log")" -eq 8
if grep -q 'SELF-CONSISTENCY FAIL' "$RAW/final-run-spec.log"; then
    echo "run-spec reported SELF-CONSISTENCY FAIL" >&2
    exit 1
fi

gpu_state
echo "capture-end $(date -u +%FT%TZ) result=PASS"
