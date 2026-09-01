#!/bin/bash
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/corpus
# Own-gen rank corpus (DRAFT-REGIME law 1). FULL-VOCAB draft head by construction, and the
# perf/gate prompts are held OUT of the pack. greedy 256/prompt (loop-law cap) + sampled
# 512/prompt x 4 vendor-default seeds over 55 prompts / 14 classes = 126,720 tokens before
# EOS truncation, which clears law 1s >=4x-topN floor for any N up to ~31k.
# --owngen-corpus-out makes the run resumable: rerun the same command to continue.
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/corpus --label owngen-nvfp4 \
  --spec-k 5 \
  --owngen ~/realgate/mtp9/corpus-prompts.tsv --owngen-out ~/realgate/mtp9/ranks-owngen.txt \
  --owngen-corpus-out ~/realgate/mtp9/corpus-ids.tsv \
  --owngen-greedy 256 --owngen-sampled 512 --owngen-seeds 4
