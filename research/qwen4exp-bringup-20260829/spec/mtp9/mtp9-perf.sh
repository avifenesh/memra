#!/bin/bash
# mtp9 Run A — the claim: the trim WIDTH table, the interleaved trim A/B, the verify
# scan-graph A/B, and the rule gates at the trimmed config.
#   $1 = trim width N (0 = every id in the ranks file = 5538, the corpus distinct set)
# Sweep widths are chosen from the measured coverage table (1024 -> 80.9%, 2048 -> 90.3%,
# 4096 -> 97.5%, 5538 -> 1.000): the qwen38 >=99.5% coverage law lands on the full set, and
# the lower rungs are there to show what coverage costs in ACCEPT and what it buys in speed.
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
N=${1:-0}
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/perf
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/perf --label trim$N-nvfp4 \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
  --spec-k 5 --spec-ab 5x256 \
  --draft-trim ~/realgate/mtp9/ranks-owngen.txt --draft-trim-n $N \
  --trim-sweep 1024,2048,3072,4096,5538 \
  --trim-ab 5x256 --vgraph-ab 5x256 \
  --verify-bit-gate 24 --spec-gate 64 --decode-timing 40 --max-new 8
