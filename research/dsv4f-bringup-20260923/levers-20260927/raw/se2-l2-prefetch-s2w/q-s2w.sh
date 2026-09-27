#!/usr/bin/env bash
# q-s2w.sh (second SE pair): probe, dense-fast blocks prefetch their weight rows into L2 before
# the PDL wait ($1, main plus one commit). Long gate hash, replay ms/token M P P M M P, against
# bin-M (main, q-s2v).
set -u
PSHA=$1
S=/root/rcpt/q-s2w.summary; R=/root/rcpt/l2-prefetch-s2w; mkdir -p $R
until grep -q S2V_DONE /root/rcpt/q-s2v.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-l2-prefetch-20260927
git worktree add -f /root/lane/t-l2 $PSHA > /dev/null 2>&1; git -C /root/lane/t-l2 checkout -q --detach $PSHA
[[ "$(git -C /root/lane/t-l2 rev-parse HEAD)" == "$PSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-l2 /root/lane/target-hm l2
mkdir -p /root/lane/bin-P; cp /root/lane/target-hm/release/dsv4_tp_replay_long_gate /root/lane/target-hm/release/memra-server /root/lane/bin-P/
echo "build P $(git -C /root/lane/t-l2 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-l2.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-M/* /root/lane/bin-P/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
i=0
for a in M P P M M P; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d /root/lane/bin-$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "S2W_DONE $PSHA" >> $S
