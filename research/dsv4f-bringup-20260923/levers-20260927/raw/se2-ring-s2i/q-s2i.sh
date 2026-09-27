#!/usr/bin/env bash
# q-s2i.sh (second SE pair): the MoE stream lane $1 (gate and up in one launch, the multi-row
# visitor on the PDL chain) and a ring-depth sweep of its visitors (KQS_STAGES 4, 8, 12; the
# committed default is 4 and the other two are box-local builds with only that define changed).
# Screening: the long gate's program hash must equal main's, and its replay ms/token per build,
# order M S4 S8 S12 S12 S8 S4 M (three timed reps inside each run).
set -u
LSHA=$1; MSHA=$2
S=/root/rcpt/q-s2i.summary; R=/root/rcpt/moe-stream-s2i; mkdir -p $R
until grep -q S2H_DONE /root/rcpt/q-s2h.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-moe-stream-20260926
git -C /root/lane/t-main checkout -q --detach $MSHA
git worktree add -f /root/lane/t-moes $LSHA > /dev/null 2>&1; git -C /root/lane/t-moes checkout -q --detach $LSHA
F=/root/lane/t-moes/crates/memra-engine/cu/moe_f16_grouped.cu
bash /root/box/build.sh /root/lane/t-main /root/lane/target-main main2
echo "build main2 $(git -C /root/lane/t-main rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-main2.log | tr '\n' ' ')" >> $S
for st in 4 8 12; do
  [[ -d /root/lane/target-moes$st ]] || cp -a /root/lane/target-main /root/lane/target-moes$st
  git -C /root/lane/t-moes checkout -q -- crates/memra-engine/cu/moe_f16_grouped.cu
  sed -i "s/^#define KQS_STAGES 4$/#define KQS_STAGES $st/" $F
  git -C /root/lane/t-moes diff > $R/variant-s$st.diff
  bash /root/box/build.sh /root/lane/t-moes /root/lane/target-moes$st moes$st
  echo "build moes$st $(grep -hE 'EXIT' /root/build-moes$st.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-s$st.diff)" >> $S
done
git -C /root/lane/t-moes checkout -q -- crates/memra-engine/cu/moe_f16_grouped.cu
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
bin() { case $1 in M) echo /root/lane/target-main/release ;; S4) echo /root/lane/target-moes4/release ;; S8) echo /root/lane/target-moes8/release ;; S12) echo /root/lane/target-moes12/release ;; esac; }
for a in M S4 S8 S12; do sha256sum $(bin $a)/dsv4_tp_replay_long_gate $(bin $a)/memra-server; done > $R/binaries.sha256
i=0
for a in M S4 S8 S12 S12 S8 S4 M; do
  i=$((i+1)); d=$R/long-r$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long r$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "S2I_DONE $LSHA" >> $S
