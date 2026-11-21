#!/usr/bin/env bash
# GB10 box, step 10: plain vs speculative greedy on the GPU route (identity, acceptance), then a decode reading.
set -u
D=$HOME/hy4flash/pr10
cd $D
mkdir -p gen
for pid in "$@"; do waitfor --gone $pid --timeout 60m --every 20s --quiet; done
free -g | head -2
S=$D/target/release/hy4f-spec
R=$HOME/hy4flash/pr7/requests
$S gen $HOME/hy4flash/hf-stage $R/chat.json 128 > gen/chat-k5.json 2> gen/chat-k5.log; echo "chat rc=$?"
$S gen $HOME/hy4flash/hf-stage $R/long.json 64 --prefill-chunk 2048 > gen/long-k5.json 2> gen/long-k5.log; echo "long rc=$?"
$S gen $HOME/hy4flash/hf-stage $R/chat.json 128 --bench 3 > gen/chat-k5-bench.json 2> gen/chat-k5-bench.log; echo "chat bench rc=$?"
echo GEN-CHAIN-DONE
