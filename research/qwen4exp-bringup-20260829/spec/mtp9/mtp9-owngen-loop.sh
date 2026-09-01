#!/bin/bash
# Own-gen corpus in BOUNDED CHUNKS. Each pass is a fresh process (clean device allocator);
# the resume ledger makes passes additive, so a pass that dies still banks its rows.
#   greedy 256/prompt (loop-law cap) + sampled 384/prompt x $SEEDS vendor-default seeds.
# 384 not 512: the longest prompt in the pack is 724 tokens and a trunk state + draft state
# + workspaces at prompt+tokens+k+4 = 1245 does NOT fit the ~2.6 GiB post-load headroom
# (measured: clean process, first generation, OOM), while 1196 did. 1117 has margin.
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
CHUNK=${1:-30}
SEEDS=${2:-2}
PACK=$HOME/realgate/mtp9/corpus-prompts-ext.tsv
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/corpus
for pass in $(seq 1 40); do
  before=$(wc -l < ~/realgate/mtp9/corpus-ids.tsv 2>/dev/null || echo 0)
  echo "=== pass $pass (ledger rows before: $before) ==="
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/corpus \
    --label owngen-p$pass --spec-k 5 \
    --owngen $PACK --owngen-out ~/realgate/mtp9/ranks-owngen.txt \
    --owngen-corpus-out ~/realgate/mtp9/corpus-ids.tsv \
    --owngen-greedy 256 --owngen-sampled 384 --owngen-seeds $SEEDS \
    --owngen-limit $CHUNK --owngen-max-prompt ${3:-400} || echo "pass $pass exited non-zero (ledger keeps what it banked)"
  after=$(wc -l < ~/realgate/mtp9/corpus-ids.tsv 2>/dev/null || echo 0)
  echo "=== pass $pass done (ledger rows after: $after) ==="
  if [ "$after" = "$before" ]; then echo "no new generations — corpus complete"; break; fi
done
