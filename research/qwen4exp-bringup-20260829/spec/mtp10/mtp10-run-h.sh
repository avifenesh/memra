#!/bin/bash
# mtp10 Run H - the close-out rule-gate battery at the MERGED branch tip, at the final
# recommended config (dev1 + adapt k_lo=1 + pmin 0.3, trim OFF): tiny fixture arms +
# verify-bit + spec-gate + greedy + interleaved A/B + the vendor-default sampled probe.
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git fetch origin && git reset --hard origin/qwen4exp-bringup-20260829 >/dev/null && git log -1 --format="HEAD %H"
cargo build --release -p memra-engine --bin qwen4exp_real_gate --bin qwen4exp_gpu_gate 2>&1 | tail -1
sha256sum target/release/qwen4exp_real_gate target/release/qwen4exp_gpu_gate
mkdir -p ~/realgate/mtp10/final
target/release/qwen4exp_gpu_gate ~/realgate/mtp10/final/tiny-fixture-gate-mtp10-final.tsv
echo "=== H1: final config, raw prompts (gates + A/B + sampled) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/final --label final-raw \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --spec-ab 5x256 --spec-sampled --verify-bit-gate 24 --spec-gate 64 --max-new 8
echo "=== H2: final config, thinkon (the shape the lane was opened on) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/final --label final-thinkon \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --spec-ab 5x256 --spec-sampled --spec-gate 64 --max-new 8
echo "=== H DONE ==="
