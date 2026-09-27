#!/usr/bin/env bash
# q-s2k.sh (second SE pair): the fused TP/EP MoE lane $1's ring geometry. The pair streams each
# warp's 8 weight rows KC k at a time through a STAGES-deep cp.async ring (committed 256, 2).
# Box-local builds change only that constexpr: G = (256, 3), H = (512, 2). Screening on the long
# gate: program hash equal to main's, replay ms/token, order M F G H H G F M.
set -u
FSHA=$1
S=/root/rcpt/q-s2k.summary; R=/root/rcpt/moe-fused-ring-s2k; mkdir -p $R
until grep -q S2J_DONE /root/rcpt/q-s2j.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
F=/root/lane/t-mft/crates/memra-engine/cu/dsv4_gpu.cu
git -C /root/lane/t-mft checkout -q --detach $FSHA
# One scratch target for both variants; only each build's long-gate binary is kept.
[[ -d /root/lane/target-mftV ]] || cp -a /root/lane/target-main /root/lane/target-mftV
for v in G:256:3 H:512:2; do
  n=${v%%:*}; kc=$(echo $v | cut -d: -f2); st=$(echo $v | cut -d: -f3)
  git -C /root/lane/t-mft checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  sed -i "s/^constexpr int DSV4_MOE_FUSED_WARPS = 4, DSV4_MOE_FUSED_KC = 256, DSV4_MOE_FUSED_STAGES = 2;$/constexpr int DSV4_MOE_FUSED_WARPS = 4, DSV4_MOE_FUSED_KC = $kc, DSV4_MOE_FUSED_STAGES = $st;/" $F
  git -C /root/lane/t-mft diff > $R/variant-$n.diff
  bash /root/box/build.sh /root/lane/t-mft /root/lane/target-mftV mft$n
  mkdir -p /root/lane/bin-mft$n && cp /root/lane/target-mftV/release/dsv4_tp_replay_long_gate /root/lane/bin-mft$n/
  echo "build mft$n KC=$kc STAGES=$st $(grep -hE 'EXIT' /root/build-mft$n.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-$n.diff)" >> $S
done
rm -rf /root/lane/target-mftV
git -C /root/lane/t-mft checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
bin() { case $1 in M) echo /root/lane/target-main/release ;; F) echo /root/lane/target-mft/release ;; *) echo /root/lane/bin-mft$1 ;; esac; }
for a in M F G H; do sha256sum $(bin $a)/dsv4_tp_replay_long_gate; done > $R/binaries.sha256
i=0
for a in M F G H H G F M; do
  i=$((i+1)); d=$R/long-r$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long r$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "S2K_DONE $FSHA" >> $S
