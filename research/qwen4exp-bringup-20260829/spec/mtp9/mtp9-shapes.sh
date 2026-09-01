#!/bin/bash
# mtp9 Run F - the PROMPT-SHAPE acceptance cells (residual item 2). Same four held-out tasks
# in three template shapes, full-vocab draft, shipped defaults.
set -e
set -o pipefail   # a piped python failure must NOT be swallowed (it was, once)
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/shapes
python3 research/qwen4exp-bringup-20260829/spec/make-shape-prompts.py ~/data/q48fn-nvfp4 ~/realgate/mtp9/shapes 2>&1 | grep -v -i "^\[transformers\]"
for shape in thinkon thinkoff efflow; do
  test -s ~/realgate/mtp9/shapes/$shape-prompts.tsv || { echo "MISSING $shape prompts"; exit 1; }
done
for shape in thinkon thinkoff efflow; do
  echo "=== shape $shape ==="
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/shapes --label shape-$shape \
    --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/$shape-prompts.tsv \
    --spec-k 5 --spec-ab 5x256 --spec-gate 64 --max-new 8
done
