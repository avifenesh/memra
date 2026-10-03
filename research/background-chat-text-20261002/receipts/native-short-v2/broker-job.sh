#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 "$PWD/tools/background-chat-text-gate.py" --phase short --out "$PWD/research/background-chat-text-20261002/receipts/native-short-v2" --binary "$PWD/target/B914/release/examples/background_accounting_gate" --model /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf --model-sha 52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de --metadata /home/avifenesh/.local/state/memra-rig-darklanes-20261002/scratch/B/vendor-profile.toml --port 18120
