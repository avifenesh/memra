#!/bin/bash
# Wait for the composed-pack pass to finish, then run the EXTENDED pack (composed 0..54 +
# the 48 owner SXC prompts at 55..102). Composed indices are byte-identical between the two
# files, so the resume ledger keeps every generation already banked and only the owner
# prompts are generated.
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
while pgrep -f "release/qwen4exp_real_gate" > /dev/null; do sleep 30; done
echo "=== composed pass finished, starting the extended pass ==="
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/corpus
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/corpus --label owngen-ext-nvfp4 \
  --spec-k 5 \
  --owngen ~/realgate/mtp9/corpus-prompts-ext.tsv --owngen-out ~/realgate/mtp9/ranks-owngen.txt \
  --owngen-corpus-out ~/realgate/mtp9/corpus-ids.tsv \
  --owngen-greedy 256 --owngen-sampled 512 --owngen-seeds 4
