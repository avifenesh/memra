#!/usr/bin/env bash
# q-v3u.sh (2x RTX PRO 6000 Server Edition), after q-v3t.sh: memra #710 greedy full-token replay.
# Lane lane/dsv4-tp-replay-capacity-20260924 12a0517b0. Long replay gate, stream MoE, greedy
# (DSV4_REPLAY_GATE_SAMPLING=greedy): cap 1024 x 304 steps with timing, cap 4096 x 1000 steps.
while ! grep -q V3T_DONE /root/rcpt/q-v3t.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3u.summary; R=/root/rcpt/tpgreedy; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-replay-capacity-20260924
git worktree add -f /root/lane/t-tpg2 12a0517b0 > /dev/null 2>&1; git -C /root/lane/t-tpg2 checkout -q --detach 12a0517b0
[[ -d /root/lane/target-tpg2 ]] || cp -a /root/lane/target-tpc /root/lane/target-tpg2
( cd /root/lane/t-tpg2 && CARGO_TARGET_DIR=/root/lane/target-tpg2 cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-tpg2/release/dsv4_tp_replay_long_gate
echo "build tpg2 $(git -C /root/lane/t-tpg2 rev-parse --short HEAD) gate $(sha256sum $X | cut -c1-16)" >> $S
for run in 1024:304 4096:1000; do
  cap=${run%%:*}; steps=${run##*:}; d=$R/c$cap-s$steps
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  ( export DSV4_REPLAY_GATE_CAPACITY=$cap DSV4_REPLAY_GATE_SAMPLING=greedy NVIDIA_TF32_OVERRIDE=0
    /root/box/gate.sh $d $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt $steps )
  echo "greedy cap $cap steps $steps $(grep -hE 'SAMPLING|REPLAY_VARIANTS|FIRST DIVERGENCE|PASS:|TIME|FAILED|panicked|GATE_DONE' $d/gate.log | sed -E 's/tokens_sha256=([0-9a-f]{16})[0-9a-f]*/tokens_sha256=\1/' | cut -c1-230 | tr '\n' ' ')" >> $S
done
echo V3U_DONE >> $S
