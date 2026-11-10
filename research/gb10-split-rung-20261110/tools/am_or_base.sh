#!/usr/bin/env bash
# GB10 unit: main's engine, Ornith's batch margin arm on the 32,774-token straddle prompt.
set -u
cd ~/.cache/memra-split-rung
CARGO_TARGET_DIR=$PWD/kt-base bash src-72368d39c4/tools/argmax-margin-gate.sh \
  ~/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf \
  --prompt $PWD/prompts/deep-32774.txt --decode batch --logdir $PWD/cells/am-or-32774-b-base/gate
