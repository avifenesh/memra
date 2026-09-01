#!/usr/bin/env bash
set -euo pipefail

if [[ ${MEMRA_DSPARK0_GPU_LOCK_HELD:-0} != 1 ]]; then
    exec flock /tmp/memra-gpu.lock env MEMRA_DSPARK0_GPU_LOCK_HELD=1 "$0" "$@"
fi

if [[ $# != 2 ]]; then
    echo "usage: $0 <label> <prompt.txt>" >&2
    exit 2
fi

label=$1
prompt=$2
root=$(cd "$(dirname "$0")/../.." && pwd)
raw="$root/research/dspark0-20260811/raw"
bin="$root/target/release/run-spec"
model=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
artifact_dir=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/dspark0-20260811

declare -A drafts=(
    [reference]="$artifact_dir/reference-byte-verbatim.gguf"
    [converter]="$artifact_dir/converter-q8-full.gguf"
)

mkdir -p "$raw"
date -Is
echo "label=$label prompt=$prompt"
nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,power.draw,memory.used,memory.total \
    --format=csv,noheader

for arm in reference converter; do
    log="$raw/parity-${label}-${arm}.log"
    echo "RUN arm=$arm draft=${drafts[$arm]} log=$log"
    env \
        CUDA_VISIBLE_DEVICES=0 \
        MEMRA_FAST=1 \
        MEMRA_MTP_DRAFT="${drafts[$arm]}" \
        MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 \
        MEMRA_NGEN=128 \
        MEMRA_SPEC_K=3 \
        MEMRA_SPEC_TEMP=0 \
        "$bin" "$model" 2>&1 | tee "$log"
done

echo "PARSED $label"
for arm in reference converter; do
    echo "[$arm]"
    rg 'text prompt|acceptance:|SELF-CONSISTENCY' "$raw/parity-${label}-${arm}.log"
done
date -Is
