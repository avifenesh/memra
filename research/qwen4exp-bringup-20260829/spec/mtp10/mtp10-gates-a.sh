#!/bin/bash
# mtp10 Run A - rule gates at the SHIPPED defaults (single card, no admission armed):
# tiny fixture arms + verify-bit + spec-gate + greedy + spec-ab 5x256 (refactor
# perf-neutrality vs the mtp9 119.97 receipt).
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate target/release/qwen4exp_gpu_gate
mkdir -p ~/realgate/mtp10/tiny ~/realgate/mtp10/defaults
target/release/qwen4exp_gpu_gate ~/realgate/mtp10/tiny/tiny-fixture-gate-mtp10.tsv
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/defaults --label mtp10-defaults \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
  --spec-k 5 --spec-ab 5x256 --spec-sampled \
  --verify-bit-gate 24 --spec-gate 64 --max-new 8
