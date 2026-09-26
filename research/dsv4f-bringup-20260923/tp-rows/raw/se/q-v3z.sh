#!/usr/bin/env bash
# q-v3z.sh (2x RTX PRO 6000 Server Edition): memra #710 TP/EP B-row, lane
# lane/dsv4-tp-rows-20260925 5d21db2a3 (stacked on the flip lane 275f83a75).
#   1. dsv4_rows_gate on TP/EP: four sessions solo vs B-row (join/leave/row moves), replay then
#      B-row while armed then replay again, and the one-row vs B-row timing.
#   2. Regressions of the walk refactor: the long replay gate (plain-304) and the DSpark gate on
#      TP/EP (verify rounds ride the same walk), proposal shas against the #720 receipts.
#   3. Served A/B: the TP/EP default (two lanes, B-row) against TP/EP one lane
#      (MEMRA_DSV4_SESSIONS=1, the flip's program) and PP-2 (`pp`), cells c1/c2/c4, one boot per row.
set -u
S=/root/rcpt/q-v3z.summary; R=/root/rcpt/tp-rows; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-20260925
git worktree add -f /root/lane/t-rows 5d21db2a3 > /dev/null 2>&1; git -C /root/lane/t-rows checkout -q --detach 5d21db2a3
[[ -d /root/lane/target-rows ]] || cp -a /root/lane/target-flip /root/lane/target-rows
bash /root/box/build.sh /root/lane/t-rows /root/lane/target-rows rows
X=/root/lane/target-rows/release
echo "build rows $(git -C /root/lane/t-rows rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-rows.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
while ! grep -q V3Y_DONE /root/rcpt/q-v3y.summary 2>/dev/null; do sleep 30; done
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { # dir cmd...
  local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 "$@" 9>&- > $d/gate.log 2>&1 )
}
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $X/dsv4_rows_gate $M /root/box/tape-rebuild.txt 24 64
echo "rows-tpep rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked|WIDTHS' $R/rows-tpep/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/long-304 $X/dsv4_tp_replay_long_gate $M /root/box/tape-rebuild.txt 304
echo "long-304 rc=$? $(grep -hE 'PASS:|FAILED|FIRST|panicked|TIME rep=2' $R/long-304/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1
  /root/box/gate.sh $d $X/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|panicked|REFUSE|out of memory|GATE_DONE' $d/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
arm_env() {
  case $1 in
    Rw) echo "MEMRA_ENV_AUDIT=on" ;;
    R4) echo "MEMRA_DSV4_SESSIONS=4" ;;
    T1) echo "MEMRA_DSV4_SESSIONS=1" ;;
    Pp) echo "MEMRA_DSV4_TOPOLOGY=pp" ;;
  esac
}
row() {
  local i=$1 arm=$2 cells=$3; local d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  timeout -k 30 2700 /root/box/cell.sh $d $X/memra-server $cells $(arm_env $arm) > $d.out 2>&1
  local rc=$?
  echo "serve r$i $arm rc=$rc $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') notarmed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hoE '[0-9]+ serving lane\(s\)[^,]*' $d/serve.log 2>/dev/null | head -1) $(grep -hE 'FATAL|engine-error|B-row' $d/serve.log 2>/dev/null | grep -v 'B-row steps up to' | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
  if [[ $rc == 124 || $rc == 137 ]]; then nvidia-smi > $d.hang-nvsmi.txt 2>&1; pkill -9 -x memra-server; sleep 20; fi
}
i=0
for arm in Rw T1 Pp Rw T1 Pp Rw R4; do i=$((i+1)); row $i $arm /root/box/cells-conc.txt; done
echo V3Z_DONE >> $S
