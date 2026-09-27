#!/usr/bin/env bash
# q-v6a.sh (SE pair): dense-fast FP8 rows per block, on the fused TP/EP MoE lane. Lane $1
# parameterizes the rows (defaults 2 and 2). Box-local builds change only the two defines:
# A = (4, 4), B = (8, 4), C = (8, 2). Screening: the long gate's program hash must stay main's;
# replay ms/token, order Z A B C C B A Z (Z = the lane as committed).
set -u
LSHA=$1
S=/root/rcpt/q-v6a.summary; R=/root/rcpt/dense-rows-v6a; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-dense-rows-20260927
git worktree add -f /root/lane/t-drows $LSHA > /dev/null 2>&1; git -C /root/lane/t-drows checkout -q --detach $LSHA
F=/root/lane/t-drows/crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh
[[ -d /root/lane/target-drows ]] || cp -a /root/lane/target-mft /root/lane/target-drows
for v in Z:2:2 A:4:4 B:8:4 C:8:2; do
  n=${v%%:*}; r1=$(echo $v | cut -d: -f2); rm_=$(echo $v | cut -d: -f3)
  git -C /root/lane/t-drows checkout -q -- crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh
  sed -i "s/^#define DSV4_DENSE_FAST_ROWS 2$/#define DSV4_DENSE_FAST_ROWS $r1/; s/^#define DSV4_DENSE_FAST_ROWS_M 2$/#define DSV4_DENSE_FAST_ROWS_M $rm_/" $F
  git -C /root/lane/t-drows diff > $R/variant-$n.diff
  bash /root/box/build.sh /root/lane/t-drows /root/lane/target-drows drows$n
  mkdir -p /root/lane/bin-drows$n && cp /root/lane/target-drows/release/dsv4_tp_replay_long_gate /root/lane/target-drows/release/memra-server /root/lane/bin-drows$n/
  echo "build drows$n ROWS=$r1 ROWS_M=$rm_ $(grep -hE 'EXIT' /root/build-drows$n.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-$n.diff)" >> $S
done
git -C /root/lane/t-drows checkout -q -- crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
for a in Z A B C; do sha256sum /root/lane/bin-drows$a/dsv4_tp_replay_long_gate; done > $R/binaries.sha256
i=0
for a in Z A B C C B A Z; do
  i=$((i+1)); d=$R/long-r$i-$a
  locked $d /root/lane/bin-drows$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long r$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "V6A_DONE $LSHA" >> $S
