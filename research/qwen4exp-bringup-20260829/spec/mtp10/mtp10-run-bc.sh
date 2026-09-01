#!/bin/bash
# mtp10 Runs B+C - dev1 placement gates + long prompts + shape round-cost + traces.
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp10/dev1 ~/realgate/mtp10/long ~/realgate/mtp10/shapes ~/realgate/mtp10/trace

echo "=== B1: dev1 rule gates (draft parity on card 1, verify-bit, spec-gate, spec-ab, sampled) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/dev1 --label mtp10-dev1 \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv --mtp-dev1 --draft-gate \
  --spec-k 5 --spec-ab 5x256 --spec-sampled --verify-bit-gate 24 --spec-gate 64 --max-new 8

echo "=== B2: dev1 long prompts (the mtp9 OOM set, 502-724 tokens) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/long --label mtp10-long \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp10/long-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-ab 3x256 --spec-gate 128 --max-new 8

echo "=== C1: shape round-cost cells (plain step vs chain/replay/verify per round) ==="
for shape in thinkon thinkoff efflow; do
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/shapes --label rc-$shape \
    --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/$shape-prompts.tsv --mtp-dev1 \
    --spec-k 5 --spec-ab 5x256 --spec-gate 64 --max-new 8
done

echo "=== C2: decay-diagnosis traces (256 tokens, all 4 prompts per shape) ==="
for shape in thinkon thinkoff; do
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/trace --label trace-$shape \
    --prompts ~/realgate/mtp9/shapes/$shape-prompts.tsv --mtp-dev1 \
    --spec-k 5 --spec-trace 256 --max-new 8
done
echo "=== BC DONE ==="
