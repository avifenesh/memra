#!/usr/bin/env bash
# q-v3o.sh (2x RTX PRO 6000 Server Edition), after q-v3n.sh: B-row identity gate after the merge of
# main into lane/dsv4-brow-serve-20260924 (1cd2cd58c): dsv4_rows_gate, solo vs B-row and pipelined
# groups, full logits bits per step.
while ! grep -q V3N_DONE /root/rcpt/q-v3n.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3o.summary; R=/root/rcpt/rowsgate2; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-serve-20260924
git worktree add -f /root/lane/t-brow2 1cd2cd58c > /dev/null 2>&1; git -C /root/lane/t-brow2 checkout -q --detach 1cd2cd58c
[[ -d /root/lane/target-brow2 ]] || cp -a /root/lane/target-rows /root/lane/target-brow2
bash /root/box/build.sh /root/lane/t-brow2 /root/lane/target-brow2 brow2
X=/root/lane/target-brow2/release
echo "build brow2 $(git -C /root/lane/t-brow2 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-brow2.log | tr '\n' ' ') gate $(sha256sum $X/dsv4_rows_gate | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
/root/box/gate.sh $R/gate $X/dsv4_rows_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64
echo "rows gate $(grep -hE 'IDENTICAL|DIVERGENCE|PASS|FAIL|PIPELINED|GATE_DONE' $R/gate/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
echo V3O_DONE >> $S
