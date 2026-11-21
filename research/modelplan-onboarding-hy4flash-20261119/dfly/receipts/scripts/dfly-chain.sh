#!/usr/bin/env bash
# GB10 box, step 10: drafter oracle and reference runs on the long dump, band backends, red arms.
set -u
D=$HOME/hy4flash/pr10
cd $D
LONG_ANCH="stride:16,range:2040-2110,at:163,at:426,at:756,at:1083"
waitfor --file $D/target/long/target.json --timeout 90m --every 60s --quiet || { echo "no long dump"; exit 1; }
R=$D/target/release/hy4f_dfly
ref() { # tag req args...
  local tag=$1 req=$2; shift 2
  systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $R checkpoint $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/$req.json target/$req rtapes/$tag "$@" \
    > $HOME/hy4flash/logs/pr10-ref-$tag.log 2>&1; echo "ref $tag rc=$?"
}
./run-dfly.sh long-f64 long f64 --anchors "$LONG_ANCH" &
ref long long --anchors "$LONG_ANCH"
wait
./run-dfly.sh chat-bf16x2af chat bf16x2a --force-drafts dtapes/chat-f64 &
./run-dfly.sh long-bf16x2af long bf16x2a --anchors "$LONG_ANCH" --force-drafts dtapes/long-f64
wait
for r in layers width no-correction topk; do
  ref long-red-$r long --anchors "$LONG_ANCH" --red $r
  ref chat-red-$r chat --red $r
done
echo DFLY-CHAIN-DONE
