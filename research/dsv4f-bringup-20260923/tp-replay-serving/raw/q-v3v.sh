#!/usr/bin/env bash
# q-v3v.sh (2x RTX PRO 6000 Server Edition), after q-v3u.sh: memra #710 full-token replay with the
# served route and mirror checks on (DSV4_REPLAY_GATE_VALIDATION=on, the default). Lane
# lane/dsv4-tp-replay-capacity-20260924 b895160d5. Long replay gate, stream MoE: default sampling
# cap 1024 x 304 (timing), greedy cap 1024 x 304 (timing), default sampling cap 4096 x 1000.
while ! grep -q V3U_DONE /root/rcpt/q-v3u.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3v.summary; R=/root/rcpt/tpval; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-replay-capacity-20260924
git worktree add -f /root/lane/t-tpv b895160d5 > /dev/null 2>&1; git -C /root/lane/t-tpv checkout -q --detach b895160d5
[[ -d /root/lane/target-tpv ]] || cp -a /root/lane/target-tpg2 /root/lane/target-tpv
( cd /root/lane/t-tpv && CARGO_TARGET_DIR=/root/lane/target-tpv cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-tpv/release/dsv4_tp_replay_long_gate
echo "build tpv $(git -C /root/lane/t-tpv rev-parse --short HEAD) gate $(sha256sum $X | cut -c1-16)" >> $S
for run in default:1024:304 greedy:1024:304 default:4096:1000; do
  samp=${run%%:*}; rest=${run#*:}; cap=${rest%%:*}; steps=${rest##*:}; d=$R/$samp-c$cap-s$steps
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  ( export DSV4_REPLAY_GATE_CAPACITY=$cap DSV4_REPLAY_GATE_SAMPLING=$samp NVIDIA_TF32_OVERRIDE=0
    /root/box/gate.sh $d $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt $steps )
  echo "val $samp cap $cap steps $steps $(grep -hE 'VALIDATION|SAMPLING|REPLAY_VARIANTS|FIRST DIVERGENCE|PASS:|TIME|FAILED|panicked|GATE_DONE' $d/gate.log | sed -E 's/tokens_sha256=([0-9a-f]{16})[0-9a-f]*/tokens_sha256=\1/' | cut -c1-230 | tr '\n' ' ')" >> $S
done
echo V3V_DONE >> $S
