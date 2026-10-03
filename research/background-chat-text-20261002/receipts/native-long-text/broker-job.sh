#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 "$PWD/tools/background-chat-text-gate.py" --phase long-text --out "$PWD/research/background-chat-text-20261002/receipts/native-long-text" --binary "$PWD/target/B914/release/examples/background_accounting_gate" --model /data/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf --model-sha 0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c --metadata /home/avifenesh/.local/state/memra-rig-darklanes-20261002/scratch/B/vendor-profile.toml --port 18125
