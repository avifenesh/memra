#!/usr/bin/env bash
# q-v3p.sh (2x RTX PRO 6000 Server Edition): memra #710 TP/EP long replay gate. Lane
# lane/dsv4-tp-graph-20260924 1c23caccf (replay armed up to min(capacity, 16384) positions):
# eager full-token program against the armed replay from position 400 over 304 steps, every
# step bit for bit (token, logits, TP/EP cache and hidden digests), replay counters.
set -u
S=/root/rcpt/q-v3p.summary; R=/root/rcpt/tpreplay; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-graph-20260924
git worktree add -f /root/lane/t-tpg 1c23caccf > /dev/null 2>&1; git -C /root/lane/t-tpg checkout -q --detach 1c23caccf
[[ -d /root/lane/target-tpg ]] || cp -a /root/lane/target-brow2 /root/lane/target-tpg
( cd /root/lane/t-tpg && CARGO_TARGET_DIR=/root/lane/target-tpg cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
echo "build tpg $(git -C /root/lane/t-tpg rev-parse --short HEAD) rc=$? gate $(sha256sum /root/lane/target-tpg/release/dsv4_tp_replay_long_gate | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/gate /root/lane/target-tpg/release/dsv4_tp_replay_long_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 304
echo "replay gate $(grep -hE 'PROTOCOL|REPLAY_VARIANTS|FIRST DIVERGENCE|PASS:|TIME|FAILED|panicked|GATE_DONE' $R/gate/gate.log | cut -c1-300 | tr '\n' ' ')" >> $S
echo V3P_DONE >> $S
