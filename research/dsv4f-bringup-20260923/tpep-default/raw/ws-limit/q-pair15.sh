#!/usr/bin/env bash
# q-pair15.sh (2x RTX PRO 6000 WS pod): memra #710 flip lane 6695e7c0b, the long replay gate
# across the served 16384 replay limit (capacity 20000, 16100 steps from position 400, greedy):
# tokens and logits bits every step, cache and hidden digests every 256 steps and on the 64
# steps either side of the handoff.
set -u
S=/root/rcpt/q-pair15.summary; R=/root/rcpt/tpep-limit; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-default-20260925
git worktree add -f /root/lane/t-flip 6695e7c0b > /dev/null 2>&1; git -C /root/lane/t-flip checkout -q --detach 6695e7c0b
[[ -d /root/lane/target-flip ]] || cp -a /root/lane/target-tpr /root/lane/target-flip
( cd /root/lane/t-flip && CARGO_TARGET_DIR=/root/lane/target-flip cargo build --release -j 56 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-flip/release/dsv4_tp_replay_long_gate
echo "build flip $(git -C /root/lane/t-flip rev-parse --short HEAD) rc=$? gate $(sha256sum $X | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
d=$R/limit16384-greedy; mkdir -p $d
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
  env NVIDIA_TF32_OVERRIDE=0 DSV4_REPLAY_GATE_CAPACITY=20000 DSV4_REPLAY_GATE_SAMPLING=greedy DSV4_REPLAY_GATE_DIGEST_EVERY=256 \
    $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 16100 9>&- > $d/gate.log 2>&1 )
echo "gate limit16384-greedy rc=$? $(grep -hE 'PASS:|FAILED|HANDOFF|FIRST|REPLAY_VARIANTS|panicked' $d/gate.log | cut -c1-260 | tr '\n' ' ')" >> $S
echo PAIR15_DONE >> $S
