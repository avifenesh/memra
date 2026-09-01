#!/usr/bin/env bash
# Current-binary baseline vs tiny NVFP4 beta+alpha dual, N=5 interleaved under one GPU lock.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
RAW=$ROOT/research/27btune-20260811/raw
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
PROMPT=$ROOT/research/e2e/prompts/p1-code-short.txt
BIN=$ROOT/target/release/run-spec

test -x "$BIN"
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

echo "capture-start $(date -u +%FT%TZ)"
echo "tree $(git -C "$ROOT" rev-parse HEAD)"
echo "binary-sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
echo "model-sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
echo "draft-sha256 $(sha256sum "$DRAFT" | cut -d' ' -f1)"
echo "contract K=3 NGEN=64 CHAT=1 prompt=$(sha256sum "$PROMPT" | cut -d' ' -f1) order=A,B,B,A,A,B,B,A,A,B"
gpu_state

common=(
    CUDA_VISIBLE_DEVICES=0
    MEMRA_MTP_DRAFT="$DRAFT"
    MEMRA_SPEC_K=3
    MEMRA_NGEN=64
    MEMRA_PROMPT_FILE="$PROMPT"
    MEMRA_CHAT=1
    MEMRA_SPEC_STATS=1
)

run_arm() {
    local rep=$1 arm=$2 label log rc
    if [[ "$arm" == A ]]; then
        label=base
    else
        label=auxdual
    fi
    log=$RAW/rp-aux-ab-r${rep}-${arm}-${label}.log
    echo "=== rep=$rep arm=$arm label=$label start=$(date -u +%FT%TZ) ==="
    gpu_state
    set +e
    if [[ "$arm" == A ]]; then
        timeout 900 env -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_DEBUG -u MEMRA_PROFILE_SPEC \
            "${common[@]}" MEMRA_NVFP4_AUX_DUAL=0 "$BIN" "$MODEL" 2>&1 | tee "$log"
    else
        timeout 900 env -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_DEBUG -u MEMRA_PROFILE_SPEC \
            "${common[@]}" "$BIN" "$MODEL" 2>&1 | tee "$log"
    fi
    rc=${PIPESTATUS[0]}
    set -e
    echo "=== rep=$rep arm=$arm label=$label end=$(date -u +%FT%TZ) rc=$rc ==="
    if (( rc != 0 )); then
        exit "$rc"
    fi
}

run_arm 1 A
run_arm 1 B
run_arm 2 B
run_arm 2 A
run_arm 3 A
run_arm 3 B
run_arm 4 B
run_arm 4 A
run_arm 5 A
run_arm 5 B

gpu_state
echo "capture-end $(date -u +%FT%TZ)"
