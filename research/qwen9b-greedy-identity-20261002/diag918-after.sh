#!/bin/bash
set -euo pipefail
python3 /home/avifenesh/.local/state/memra-rig-20261002/scratch/A/diag918-after.py
out=$PWD/research/qwen9b-greedy-identity-20261002/after
MEMRA_NGEN=64 MEMRA_PROMPT='Count from one to twenty in words, separated by commas.' target/release/run-spec /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf > "$out/run-spec-k1-8.log" 2>&1
