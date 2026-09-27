#!/usr/bin/env bash
# q-v3t.sh (2x RTX PRO 6000 Server Edition): memra #710 full-token replay capacity lift. Lane
# lane/dsv4-tp-replay-capacity-20260924 4a7d0433e (replay admits capacities up to 16384). Long replay
# gate, stream MoE: cap 4096 x 3000 steps, cap 16384 x 3000 steps (positions 400..3400, identity
# every step), then cap 1024 and cap 4096 x 600 steps for the cost of the larger indexer grid at
# equal positions.
set -u
S=/root/rcpt/q-v3t.summary; R=/root/rcpt/tpcap; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-replay-capacity-20260924
git worktree add -f /root/lane/t-tpc 4a7d0433e > /dev/null 2>&1; git -C /root/lane/t-tpc checkout -q --detach 4a7d0433e
[[ -d /root/lane/target-tpc ]] || cp -a /root/lane/target-tps /root/lane/target-tpc
( cd /root/lane/t-tpc && CARGO_TARGET_DIR=/root/lane/target-tpc cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-tpc/release/dsv4_tp_replay_long_gate
echo "build tpc $(git -C /root/lane/t-tpc rev-parse --short HEAD) gate $(sha256sum $X | cut -c1-16)" >> $S
for run in 4096:3000 16384:3000 1024:600 4096:600; do
  cap=${run%%:*}; steps=${run##*:}; d=$R/c$cap-s$steps
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  ( export DSV4_REPLAY_GATE_CAPACITY=$cap NVIDIA_TF32_OVERRIDE=0
    /root/box/gate.sh $d $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt $steps )
  echo "cap $cap steps $steps $(grep -hE 'REPLAY_VARIANTS|FIRST DIVERGENCE|PASS:|TIME|FAILED|panicked|GATE_DONE' $d/gate.log | sed -E 's/tokens_sha256=([0-9a-f]{16})[0-9a-f]*/tokens_sha256=\1/' | cut -c1-230 | tr '\n' ' ')" >> $S
done
echo V3T_DONE >> $S
