#!/usr/bin/env bash
# GB10 unit: the argmax margin gate on the real-text prompt (the first 16,390 tokens of
# research/e2e/prompts/p4-16k.txt, its window straddling 16,384). usage: am_p4.sh base|head
set -u
cd ~/.cache/memra-split-rung
case $1 in
  base) KT=kt-base; SRC=src-72368d39c4; OUT=am-27b-p4-16390-e-base ;;
  head) KT=kt-head-1a4f769861; SRC=src-1a4f769861; OUT=am-27b-p4-16390-e ;;
esac
CARGO_TARGET_DIR=$PWD/$KT bash $SRC/tools/argmax-margin-gate.sh \
  ~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
  --prompt $PWD/prompts/p4-16390.txt --logdir $PWD/cells/$OUT/gate
