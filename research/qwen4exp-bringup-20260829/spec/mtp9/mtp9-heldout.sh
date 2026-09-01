#!/bin/bash
# mtp9 Run D — the IN-CLASS held-out cell, the decisive diagnostic for the trim regression.
# Run A measured the trim on the perf prompt: a RAW continuation, while the rank corpus is
# chat-template rendered — DRAFT-REGIME law 1 says ranks inherit their corpus MIX, so that
# is the out-of-class cell by construction. These prompts are chat-shaped like the corpus but
# were HELD OUT of it (selected by text). If the trim recovers here the binding constraint is
# CLASS COVERAGE; if it still loses, it is corpus SIZE (93k tokens only ever discovered 5,538
# distinct ids = 2.2% of a 248,320 vocab).
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/heldout
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/heldout --label heldout-trim5538 \
  --prompts ~/realgate/mtp9/heldout-prompts.tsv \
  --draft-trim ~/realgate/mtp9/ranks-owngen.txt --draft-trim-n 0 \
  --spec-k 5 --spec-ab 5x256 \
  --trim-sweep 2048,4096,5538 --trim-ab 5x256 \
  --spec-gate 64 --max-new 8
