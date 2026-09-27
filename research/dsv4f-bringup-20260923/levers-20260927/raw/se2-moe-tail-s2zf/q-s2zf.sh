#!/usr/bin/env bash
# q-s2zf.sh (second SE pair): the one-launch MoE tail with the replay census fix ($1) against
# main (bin-zeM, 286c0c54c). Long gate hash, TP/EP rows, KV split, DSpark TP/EP, the long gate
# M T T M M T and served cells-pdl M T T M.
set -u
TSHA=$1
S=/root/rcpt/q-s2zf.summary; R=/root/rcpt/tail-fix-s2zf; mkdir -p $R
while ps -eo args | grep -q "[b]in-zeC/dsv4_rows_gate"; do sleep 20; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-tail-fuse-20260927
git -C /root/lane/t-zeT checkout -q --detach $TSHA
[[ "$(git -C /root/lane/t-zeT rev-parse HEAD)" == "$TSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-zeT /root/lane/target-hm zeT2
X=/root/lane/target-hm/release; mkdir -p /root/lane/bin-T2; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-T2/
echo "build T2 $(git -C /root/lane/t-zeT rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zeT2.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-T2/* /root/lane/bin-zeM/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XT=/root/lane/bin-T2
locked $R/long-304 $XT/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 T rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XT/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XT/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XT/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep T $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-zeM ;; *) echo /root/lane/bin-T2 ;; esac; }
i=0
for a in M T T M M T; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M T T M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2ZF_DONE $TSHA" >> $S
