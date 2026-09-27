#!/usr/bin/env bash
# q-v3y.sh (2x RTX PRO 6000 Server Edition): memra #710 flip lane 275f83a75, the long replay gate's
# handoff rows again (the 9d19b5014 gate read its capture count after the handoff dropped the
# graphs, so those three rows died before their verdict).
set -u
S=/root/rcpt/q-v3y.summary; R=/root/rcpt/tpep-default; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-default-20260925
git -C /root/lane/t-flip checkout -q --detach 275f83a75
( cd /root/lane/t-flip && CARGO_TARGET_DIR=/root/lane/target-flip cargo build --release -j 56 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build-y.log 2>&1
X=/root/lane/target-flip/release
echo "build flip $(git -C /root/lane/t-flip rev-parse --short HEAD) rc=$? gate $(sha256sum $X/dsv4_tp_replay_long_gate | cut -c1-16)" >> $S
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
gate() { # name steps env...
  local name=$1 steps=$2; shift 2; local d=$R/gate-$name; mkdir -p $d
  [[ -f $d/gate.log ]] && mv $d/gate.log $d/gate-9d19b5014.log
  wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
    env NVIDIA_TF32_OVERRIDE=0 "$@" $X/dsv4_tp_replay_long_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt $steps 9>&- > $d/gate.log 2>&1 )
  echo "gate $name rc=$? $(grep -hE 'PASS:|FAILED|HANDOFF|FIRST|REPLAY_VARIANTS|panicked' $d/gate.log | cut -c1-230 | tr '\n' ' ')" >> $S
}
gate handoff-greedy 500 DSV4_REPLAY_GATE_CAPACITY=2048 DSV4_REPLAY_GATE_LIMIT=640 DSV4_REPLAY_GATE_SAMPLING=greedy
gate handoff-default 500 DSV4_REPLAY_GATE_CAPACITY=2048 DSV4_REPLAY_GATE_LIMIT=640
gate limit16384-greedy 16100 DSV4_REPLAY_GATE_CAPACITY=20000 DSV4_REPLAY_GATE_SAMPLING=greedy
echo V3Y_DONE >> $S
