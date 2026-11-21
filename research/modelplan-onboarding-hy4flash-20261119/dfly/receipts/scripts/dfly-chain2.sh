#!/usr/bin/env bash
# GB10 box, step 10: the drafter's band backends (bf16x2a, forced along the f64 drafts) and the reference red arms.
set -u
D=$HOME/hy4flash/pr10
cd $D
LONG_ANCH="stride:16,range:2040-2110,at:163,at:426,at:756,at:1083"
R=$D/target/release/hy4f_dfly
ref() { # tag req args...
  local tag=$1 req=$2; shift 2
  systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $R checkpoint $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/$req.json target/$req rtapes/$tag "$@" \
    > $HOME/hy4flash/logs/pr10-ref-$tag.log 2>&1; echo "ref $tag rc=$?"
}
for pid in "$@"; do waitfor --gone $pid --timeout 60m --every 20s --quiet; done
./run-dfly.sh chat-bf16x2af chat bf16x2a --force-drafts dtapes/chat-f64
./run-dfly.sh long-bf16x2af long bf16x2a --anchors "$LONG_ANCH" --force-drafts dtapes/long-f64 &
ref chat-red-layers chat --red layers
ref chat-red-width chat --red width
ref chat-red-no-correction chat --red no-correction
ref chat-red-topk chat --red topk
wait
ref long-red-layers long --anchors "$LONG_ANCH" --red layers &
ref long-red-width long --anchors "$LONG_ANCH" --red width
wait
ref long-red-no-correction long --anchors "$LONG_ANCH" --red no-correction &
ref long-red-topk long --anchors "$LONG_ANCH" --red topk
wait
echo DFLY-CHAIN2-DONE
