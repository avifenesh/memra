#!/usr/bin/env bash
# q-s2s.sh (second SE pair): a TIMING PROBE, not a lane. How much of the fused MoE pair's time is
# its per-CTA activation mirror? Box-local builds of main $1: X1 skips the gate/up kernel's x
# mirror (zero activations), X2 skips both mirrors. Wrong outputs by design: only the long gate's
# replay ms/token is read (it compares replay with eager of the same binary). Order M X1 X2 X2 X1 M.
set -u
MSHA=$1
S=/root/rcpt/q-s2s.summary; R=/root/rcpt/mirror-probe-s2s; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main
git worktree add -f /root/lane/t-probe $MSHA > /dev/null 2>&1; git -C /root/lane/t-probe checkout -q --detach $MSHA
[[ -d /root/lane/target-probe ]] || cp -a /root/lane/target-small2 /root/lane/target-probe
F=/root/lane/t-probe/crates/memra-engine/cu/dsv4_gpu.cu
for v in M X1 X2; do
  git -C /root/lane/t-probe checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  case $v in X1) python3 /root/box/mirror_hack.py $F gu ;; X2) python3 /root/box/mirror_hack.py $F both ;; esac
  git -C /root/lane/t-probe diff > $R/variant-$v.diff
  bash /root/box/build.sh /root/lane/t-probe /root/lane/target-probe probe$v
  mkdir -p /root/lane/bin-probe$v && cp /root/lane/target-probe/release/dsv4_tp_replay_long_gate /root/lane/bin-probe$v/
  echo "build probe$v $(grep -hE 'EXIT' /root/build-probe$v.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-$v.diff)" >> $S
done
git -C /root/lane/t-probe checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
i=0
for a in M X1 X2 X2 X1 M; do
  i=$((i+1)); d=$R/long-r$i-$a
  locked $d /root/lane/bin-probe$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long r$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
rm -rf /root/lane/target-probe /root/lane/bin-probe*
echo "S2S_DONE $MSHA" >> $S
