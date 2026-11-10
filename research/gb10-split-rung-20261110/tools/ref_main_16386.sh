#!/usr/bin/env bash
# GB10 unit: main's engine on the 16,386-token prefix of deep-16390 (its last row is 16,385), eager
# arm: forward_last (f32 K/V), prime_cache (the PRIME-GATE line: prefill program over the quantized
# cache) and tokenwise decode at that row, for memra#1175.
set -u
cd ~/.cache/memra-split-rung
CARGO_TARGET_DIR=$PWD/kt-base bash src-72368d39c4/tools/argmax-margin-gate.sh \
  ~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
  --prompt $PWD/prompts/deep-16386.txt --logdir $PWD/cells/ref-main-16386b/gate
