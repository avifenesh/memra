#!/usr/bin/env bash
# q-v7f.sh (SE pair): the one-row dense-fast body with four iterations' loads in flight ($1)
# against its main base ($2). Long gate hash, TP/EP rows, DSpark TP/EP, the long gate
# M X X M M X, served cells-pdl M X X M.
set -u
XSHA=$1; MSHA=$2
S=/root/rcpt/q-v7f.summary; R=/root/rcpt/mlp4-v7f; mkdir -p $R
while ps -eo args | grep -q "[d]sv4_rows_gate"; do sleep 30; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
cd /root/lane/memra && git fetch -q origin main lane/dsv4-dense-mlp4-20260928
for v in M:$MSHA X:$XSHA; do IFS=: read n sha <<< "$v"
  git worktree add -f /root/lane/t-7f$n $sha > /dev/null 2>&1; git -C /root/lane/t-7f$n checkout -q --detach $sha
  [[ "$(git -C /root/lane/t-7f$n rev-parse HEAD)" == "$sha" ]] || { echo "TREE_MISMATCH $n" >> $S; exit 1; }
  bash /root/box/build.sh /root/lane/t-7f$n /root/lane/target-main13 7f$n
  mkdir -p /root/lane/bin-7f$n; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-7f$n/
  echo "build $n $(git -C /root/lane/t-7f$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-7f$n.log | tr '\n' ' ')" >> $S
done
sha256sum /root/lane/bin-7f*/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XX=/root/lane/bin-7fX
locked $R/long-304 $XX/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 X rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XX/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XX/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep X $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-7fM ;; *) echo /root/lane/bin-7fX ;; esac; }
i=0
for a in M X X M M X; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M X X M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
for n in M X; do git -C /root/lane/memra worktree remove --force /root/lane/t-7f$n; done; git -C /root/lane/memra worktree prune
echo "V7F_DONE $XSHA" >> $S
