#!/usr/bin/env bash
set -euo pipefail

if [[ ${MEMRA_DSPARK0_GPU_LOCK_HELD:-0} != 1 ]]; then
    exec flock /tmp/memra-gpu.lock env MEMRA_DSPARK0_GPU_LOCK_HELD=1 "$0" "$@"
fi

root=$(cd "$(dirname "$0")/../.." && pwd)
raw="$root/research/dspark0-20260811/raw"
bin="$root/target/release/run-spec"
model=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
prompt="$root/research/e2e/prompts/p1-code-short.txt"
artifact_dir=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/dspark0-20260811

declare -A drafts=(
    [reference]="$artifact_dir/reference-byte-verbatim.gguf"
    [converter]="$artifact_dir/converter-q8-full.gguf"
)

date -Is
nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,power.draw,memory.used,memory.total \
    --format=csv,noheader

for arm in reference converter; do
    log="$raw/k1-8-${arm}.log"
    echo "RUN arm=$arm draft=${drafts[$arm]} log=$log"
    env -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0 \
        MEMRA_FAST=1 \
        MEMRA_MTP_DRAFT="${drafts[$arm]}" \
        MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 \
        MEMRA_NGEN=128 \
        MEMRA_SPEC_TEMP=0 \
        "$bin" "$model" 2>&1 | tee "$log"
done

for arm in reference converter; do
    echo "PARSED [$arm]"
    rg 'generate_spec K=|acceptance:|SELF-CONSISTENCY' "$raw/k1-8-${arm}.log"
done
date -Is
