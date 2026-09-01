#!/bin/bash
# mtp9 Run G - the K ladder on the ONE shape where spec wins (thinkoff, 1.52x). The mtp7/mtp8
# knee (K=5) was found on a raw continuation whose accept is 0.840; the thinkoff shape runs
# 4.34 committed tokens/round with a different accept curve, so its knee is an open question
# and it is the shape any shape-aware spec admission would actually serve.
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/shapes
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/shapes --label thinkoff-ladder \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkoff-prompts.tsv \
  --spec-k 5 --spec-ab 5x256 --spec-ladder 3,4,5,6,7,8 --spec-sampled --max-new 8
