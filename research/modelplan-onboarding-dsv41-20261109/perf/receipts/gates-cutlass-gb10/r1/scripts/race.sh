#!/usr/bin/env bash
# GB10, #1286 item 4: the race harness at the restacked head. Each run is dsv41-gate tp --reps 40 (the gate's 12-, 16-
# and 32-row passes 40 times, round r > 0 on the prompt rows from 7r) with two ranks on the card, peer-pull, layers 2
# and 24, no decode steps: the configuration of receipts race-cutlass-gb10 (11 to 28 corrupt passes a run before the
# fix). Arms, interleaved run by run: --dense-form 2 (every dense GEMM on CUTLASS's swapped tileN-8 form), --dense-form 3
# (swapped tileN 16), and the default form; with CONTROL 1 one run first of the race's own debug build d6 at
# V41DBG_REPS=40 (the grouped swapped forms still in it: the positive control that the harness finds the race on this
# card today).
# usage: race.sh TAG [N (default 3)] [ARMS (default "2 3 d")] [CONTROL (1: the d6 run first)]   Done-file out/race-TAG/runs.txt, marker RACE-DONE.
set -u
TAG=$1; N=${2:-3}; ARMS=${3:-2 3 d}; CONTROL=${4:-0}
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$R/bins/$TAG
O=$R/out/race-$TAG; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
line() {  # NAME LOG RC SECS
  echo "$1 rc $3 $(grep -ao '[0-9]* checks, [0-9]* failed' $2 | tail -1) FAIL lines $(grep -ac '^FAIL' $2) passes $(grep -acE '^(PASS|FAIL) layer[0-9]+/rows[0-9]+(/rep[0-9]+)?(/pipelined)?/rank0/out_stream' $2) secs $4 $(date +%T)" >> $O/runs.txt
  grep -a '^FAIL' $2 | head -4 >> $O/runs.txt
}
echo "bins $(sha256sum $B/dsv41-gate | cut -c1-16) $B; 2 ranks on one card (--devices 0,0), peer-pull, layers 2,24, --reps 40" >> $O/runs.txt
if [ "$CONTROL" = 1 ] && [ -x $HOME/.cache/dsv41-cutlass/bins/d6/dsv41-gate ]; then
  t0=$(date +%s)
  timeout 8m env V41DBG_REPS=40 $HOME/.cache/dsv41-cutlass/bins/d6/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 \
    --devices 0,0 --decode 0 --token-map $TM --out $O/control-d6.json > $O/control-d6.log 2>&1
  line "control d6 (the unfixed grouped forms, V41DBG_REPS=40)" $O/control-d6.log $? $(( $(date +%s) - t0 ))
fi
for i in $(seq $N); do
  for a in $ARMS; do
    XF=""; [ $a = d ] || XF="--dense-form $a"
    name="form-$a-$i"
    t0=$(date +%s)
    timeout 8m $B/dsv41-gate $XF tp $C $D/captures-v2/needle --layers 2,24 --devices 0,0 --decode 0 --reps 40 \
      --token-map $TM --out $O/$name.json > $O/$name.log 2>&1
    line "$name (dense form $([ $a = d ] && echo default || echo $a))" $O/$name.log $? $(( $(date +%s) - t0 ))
  done
done
unset MEMRA_TP_TRANSPORT
echo "RACE-DONE $(date +%T)" >> $O/runs.txt
