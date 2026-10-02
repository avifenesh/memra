#!/bin/bash
set -euo pipefail
coord=/home/avifenesh/.local/state/memra-rig-20261002
python3 "$coord/scratch/A/review-numeric.py"
MEMRA_NGEN=64 MEMRA_PROMPT='Count from one to twenty in words, separated by commas.' target/release/run-spec /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf > research/cache-metrics-20261002/review-native/run-spec-k1-8.log 2>&1
python3 "$coord/scratch/A/review-lifecycle.py" --promtool "$coord/receipts/A/tools/promtool" --credentials-dir "$coord/scratch/A/review-credentials"
