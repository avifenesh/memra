#!/usr/bin/env bash
# GB10, #1286 item 4 strengthened (one lease): the race harness on the shipping default dense form, three arms run as
# triples with the race's own debug build d6 before each triple as the positive control:
#   plain   dsv41-gate tp --reps REPS
#   delay   the same with --side-delay NS (every second-stream branch held NS before it starts)
#   poison  the same with --poison-free (main-stream buffers filled with 0xFF as they drop)
# Two ranks on the card, peer-pull, layers 2 and 24, no decode steps (receipts race-cutlass-gb10's configuration).
# "passes" counts the compared passes (rank 0's out_stream of every round's pipelined and wide pass, 12 a round).
# usage: race2.sh TAG PART [TRIPLES (2)] [REPS (160)] [NS (2000000)]
#        -> out/race2-TAG/runs-PART.txt, logs PART-*.log; marker RACE2-PART-DONE
set -u
TAG=$1; P=$2; NT=${3:-2}; REPS=${4:-160}; NS=${5:-2000000}
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$R/bins/$TAG
CTL=$R/bins/d6/dsv41-gate
O=$R/out/race2-$TAG; mkdir -p $O; F=$O/runs-$P.txt; : > $F
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
line() {  # NAME LOG RC SECS
  echo "$1 rc $3 $(grep -ao '[0-9]* checks, [0-9]* failed' $2 | tail -1) FAIL lines $(grep -ac '^FAIL' $2) passes $(grep -acE '^(PASS|FAIL) layer[0-9]+/rows[0-9]+(/rep[0-9]+)?(/pipelined)?/rank0/out_stream' $2) secs $4 $(date +%T)" >> $F
  grep -a '^FAIL' $2 | head -4 >> $F
}
echo "part $P: bins $(sha256sum $B/dsv41-gate | cut -c1-16) $TAG, control $(sha256sum $CTL | cut -c1-16) d6; 2 ranks on one card, peer-pull, layers 2,24, dense form default, --reps $REPS, side delay $NS ns; $(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader)" >> $F
for t in $(seq $NT); do
  n=$P-$t
  t0=$(date +%s)
  timeout 8m env V41DBG_REPS=40 $CTL tp $C $D/captures-v2/needle --layers 2,24 --devices 0,0 --decode 0 --token-map $TM \
    --out $O/$n-control.json > $O/$n-control.log 2>&1
  line "$n-control (d6, the unfixed grouped forms, V41DBG_REPS=40)" $O/$n-control.log $? $(( $(date +%s) - t0 ))
  for arm in plain delay poison; do
    case $arm in plain) X="" ;; delay) X="--side-delay $NS" ;; poison) X="--poison-free" ;; esac
    t0=$(date +%s)
    timeout 12m $B/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 --devices 0,0 --decode 0 --reps $REPS $X \
      --token-map $TM --out $O/$n-$arm.json > $O/$n-$arm.log 2>&1
    line "$n-$arm (dense form default${X:+, $X})" $O/$n-$arm.log $? $(( $(date +%s) - t0 ))
  done
done
unset MEMRA_TP_TRANSPORT
echo "RACE2-$P-DONE $(date +%T)" >> $F
