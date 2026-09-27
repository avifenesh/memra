#!/usr/bin/env bash
# q-v3s.sh (2x RTX PRO 6000 Server Edition), after q-v3r.sh: memra #710 full-token replay on the
# served one-token stream MoE. Lane lane/dsv4-tp-replay-stream-20260924 8dee5e824. The long replay gate
# twice: DSV4_REPLAY_GATE_MOE=stream (the served program) and =sktail (the pinned split-K set),
# each: identity arm (replay vs eager per step) and the alternating-order timing arm. The two
# expert programs must give the same tokens sha (112c2fc6... on the split-K run of 1c23caccf).
while ! grep -q V3R_DONE /root/rcpt/q-v3r.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3s.summary; R=/root/rcpt/tpstream; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-replay-stream-20260924
git worktree add -f /root/lane/t-tps 8dee5e824 > /dev/null 2>&1; git -C /root/lane/t-tps checkout -q --detach 8dee5e824
[[ -d /root/lane/target-tps ]] || cp -a /root/lane/target-tpa /root/lane/target-tps
( cd /root/lane/t-tps && CARGO_TARGET_DIR=/root/lane/target-tps cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-tps/release/dsv4_tp_replay_long_gate
echo "build tps $(git -C /root/lane/t-tps rev-parse --short HEAD) gate $(sha256sum $X | cut -c1-16)" >> $S
for moe in stream sktail; do
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  ( export DSV4_REPLAY_GATE_MOE=$moe NVIDIA_TF32_OVERRIDE=0
    /root/box/gate.sh $R/$moe $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 304 )
  echo "moe $moe $(grep -hE 'MOE|REPLAY_VARIANTS|FIRST DIVERGENCE|PASS:|TIME|FAILED|panicked|GATE_DONE' $R/$moe/gate.log | cut -c1-260 | tr '\n' ' ')" >> $S
done
echo V3S_DONE >> $S
