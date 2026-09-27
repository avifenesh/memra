#!/usr/bin/env bash
# q-s2r.sh (second SE pair): the small-kernel lane $1 after its wq_b / indexer wq_b pairing and its
# restack on the fixed pair lane. Gates (long gate hash, TP/EP rows, DSpark TP/EP) and the long
# gate's replay ms/token S S2 S2 S (S = the lane before this commit, target-small).
set -u
SSHA=$1
S=/root/rcpt/q-s2r.summary; R=/root/rcpt/small-regs2-s2r; mkdir -p $R
until grep -q S2Q_DONE /root/rcpt/q-s2q.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-small-regs-20260927
git worktree add -f /root/lane/t-small2 $SSHA > /dev/null 2>&1; git -C /root/lane/t-small2 checkout -q --detach $SSHA
[[ "$(git -C /root/lane/t-small2 rev-parse HEAD)" == "$SSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
[[ -d /root/lane/target-small2 ]] || cp -a /root/lane/target-small /root/lane/target-small2
bash /root/box/build.sh /root/lane/t-small2 /root/lane/target-small2 small2
echo "build small2 $(git -C /root/lane/t-small2 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-small2.log | tr '\n' ' ')" >> $S
XS=/root/lane/target-small/release; X2=/root/lane/target-small2/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $X2/memra-server $X2/dsv4_tp_replay_long_gate $X2/dsv4_rows_gate $X2/dsv4-gpu-dspark-gate > $R/binaries.sha256
locked $R/long-304 $X2/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 S2 rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $X2/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $X2/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep S2 $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in S) echo $XS ;; *) echo $X2 ;; esac; }
i=0
for a in S T T S; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "S2R_DONE $SSHA" >> $S
