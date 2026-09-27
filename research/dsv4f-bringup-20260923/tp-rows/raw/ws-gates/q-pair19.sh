#!/usr/bin/env bash
# q-pair19.sh (WS pod): memra #710 B-row graphs, lane lane/dsv4-tp-rows-graph-20260926 (sha in
# the summary). dsv4_rows_gate on TP/EP: solo vs eager B-row, replay re-entry, the graph arm
# (bit identity + capture/replay counts), sampled graph vs eager draws, and timing.
set -u
SHA=$1
S=/root/rcpt/q-pair19.summary; R=/root/rcpt/rows-graph-$SHA; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-graph-20260926
git worktree add -f /root/lane/t-rg $SHA > /dev/null 2>&1; git -C /root/lane/t-rg checkout -q --detach $SHA
[[ -d /root/lane/target-rg ]] || cp -a /root/lane/target-rows /root/lane/target-rg
( cd /root/lane/t-rg && CARGO_TARGET_DIR=/root/lane/target-rg cargo build --release -j 56 -p memra-engine --bin dsv4_rows_gate ) > $R/build.log 2>&1
echo "build rg $SHA rc=$? $(grep -cE '^error' $R/build.log) errors" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
  env NVIDIA_TF32_OVERRIDE=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep timeout 3000 /root/lane/target-rg/release/dsv4_rows_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $R/gate.log 2>&1 )
echo "rows-graph rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked|Error|error' $R/gate.log | cut -c1-220 | tr '\n' ' ')" >> $S
echo "PAIR19_DONE $SHA" >> $S
