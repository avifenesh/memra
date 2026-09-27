#!/usr/bin/env bash
# q-v6c.sh (SE pair): the fused TP/EP MoE pair's CTA width, on lane $1 (fused, multi-row). With
# about three local experts, the committed 4 warps per projection put ~192 live CTAs of 8 warps on
# 188 SMs, so the SMs holding two set the kernel time. Box-local builds change only the constexpr
# line: I = (WP 2, KC 256, ST 3), J = (WP 1, KC 256, ST 3), K = (WP 2, KC 512, ST 2); F = committed
# (4, 256, 2). TP/EP only (the full-bank form's tile counters are sized for 4 warps). Screening on
# the long gate: program hash equal to main's, replay ms/token, order F I J K K J I F.
set -u
FSHA=$1
S=/root/rcpt/q-v6c.summary; R=/root/rcpt/moe-fused-cta-v6c; mkdir -p $R
until grep -q V6B_DONE /root/rcpt/q-v6b.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-fused-tpep-20260926
git -C /root/lane/t-mft checkout -q -- . ; git -C /root/lane/t-mft checkout -q --detach $FSHA
F=/root/lane/t-mft/crates/memra-engine/cu/dsv4_gpu.cu
[[ -d /root/lane/target-mftV ]] || cp -a /root/lane/target-mft /root/lane/target-mftV
for v in F:4:256:2 I:2:256:3 J:1:256:3 K:2:512:2; do
  n=${v%%:*}; wp=$(echo $v | cut -d: -f2); kc=$(echo $v | cut -d: -f3); st=$(echo $v | cut -d: -f4)
  git -C /root/lane/t-mft checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  sed -i "s/^constexpr int DSV4_MOE_FUSED_WARPS = 4, DSV4_MOE_FUSED_KC = 256, DSV4_MOE_FUSED_STAGES = 2;$/constexpr int DSV4_MOE_FUSED_WARPS = $wp, DSV4_MOE_FUSED_KC = $kc, DSV4_MOE_FUSED_STAGES = $st;/" $F
  git -C /root/lane/t-mft diff > $R/variant-$n.diff
  bash /root/box/build.sh /root/lane/t-mft /root/lane/target-mftV cta$n
  mkdir -p /root/lane/bin-cta$n && cp /root/lane/target-mftV/release/dsv4_tp_replay_long_gate /root/lane/target-mftV/release/memra-server /root/lane/bin-cta$n/
  echo "build cta$n WP=$wp KC=$kc ST=$st $(grep -hE 'EXIT' /root/build-cta$n.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-$n.diff)" >> $S
done
git -C /root/lane/t-mft checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
rm -rf /root/lane/target-mftV
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
for a in F I J K; do sha256sum /root/lane/bin-cta$a/dsv4_tp_replay_long_gate; done > $R/binaries.sha256
i=0
for a in F I J K K J I F; do
  i=$((i+1)); d=$R/long-r$i-$a
  locked $d /root/lane/bin-cta$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long r$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "V6C_DONE $FSHA" >> $S
