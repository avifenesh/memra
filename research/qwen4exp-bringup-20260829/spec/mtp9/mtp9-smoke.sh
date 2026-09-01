#!/bin/bash
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/smoke
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/smoke --label smoke \
  --prompts ~/realgate/dump/prompts.tsv --spec-k 5 \
  --owngen ~/realgate/mtp9/smoke-prompts.tsv --owngen-out ~/realgate/mtp9/smoke-ranks.txt \
  --owngen-greedy 24 --owngen-sampled 32 --owngen-seeds 2 \
  --draft-trim ~/realgate/mtp9/smoke-ranks.txt --draft-trim-n 4096 \
  --trim-ab 1x48 --max-new 8
