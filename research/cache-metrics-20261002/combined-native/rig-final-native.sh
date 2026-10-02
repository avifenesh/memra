#!/bin/bash
set -euo pipefail
coord=/home/avifenesh/.local/state/memra-rig-20261002
python3 "$coord/scratch/A/rig-final-metrics.py" --promtool "$coord/receipts/A/tools/promtool" --credentials-dir "$coord/scratch/A/credentials"
python3 "$coord/scratch/A/rig-final-formats.py" --promtool "$coord/receipts/A/tools/promtool" --credentials-dir "$coord/scratch/A/format-credentials"
python3 tools/metrics-background-gate.py --binary target/release/examples/background_accounting_gate --model /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf --out research/cache-metrics-20261002/combined-background --port 18118 --promtool "$coord/receipts/A/tools/promtool" --external-lock 9
python3 "$coord/scratch/A/rig-final-777.py"
