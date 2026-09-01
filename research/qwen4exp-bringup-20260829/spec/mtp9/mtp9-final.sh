#!/bin/bash
# mtp9 Run E - the rule-gate battery at the SHIPPED defaults (no trim armed, verify graphs
# OFF), i.e. the mtp8 program with the mtp9 code in it. Carries the serving law's sampled
# vendor-default spec-engagement receipt at the config that actually ships.
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/final ~/realgate/mtp9/tp2
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/final --label mtp9-defaults \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
  --spec-k 5 --spec-ab 5x256 --spec-ladder 3,4,5,6,7,8 --spec-sampled \
  --verify-bit-gate 24 --spec-gate 64 --decode-timing 40 --max-new 8
echo "=== tp2 rule gate ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/tp2 --label mtp9-tp2 \
  --goldens ~/realgate/dump --tp2-gate 24 --decode-timing 40
