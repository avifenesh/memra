#!/bin/bash
# wrx, inside a two-card lease (win.sh stress2; the lease's cards are devices 0 and 1): #1286 item 4, the race
# harness at one rank per card on the pair, the default dense form. Each run is dsv41-gate tp --reps 40 (the gate's
# 12-, 16- and 32-row passes 40 times, round r > 0 on the prompt rows from 7r, so a stale or wrong read returns other
# bytes), layers 2 and 24, no decode steps: the configuration that put 11 to 28 corrupt passes in a run on GB10 before
# the fix (receipts race-cutlass-gb10). N runs a transport, interleaved: peer-pull (the race's transport) and the
# default (device-push on this pair). "passes" counts the compared passes (rank 0's out_stream of every round's
# pipelined and wide pass); with no failure in N of them the per-pass rate is below 3 / N (95%).
# usage: stress2.sh BINS_TAG [N (default 4)]   -> /home/access/dsv41/cfix2/stress2/{runs.txt,<tr>-<i>.log,.json}
set -u
D=/home/access/dsv41
B=$D/perf-bins/${1:?bins tag}
N=${2:-4}
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$D/cfix2/stress2
mkdir -p $O
: > $O/runs.txt
export MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel
echo "bins $B: $(cat $B/SOURCE); dsv41-gate $(sha256sum $B/dsv41-gate | cut -c1-16); dense form: default; devices 0,1 (one rank a card)" >> $O/runs.txt
for i in $(seq $N); do
  for tr in peer-pull default; do
    if [ $tr = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$tr; fi
    t0=$(date +%s)
    timeout 10m $B/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 --devices 0,1 --decode 0 --reps 40 \
      --token-map $TM --out $O/$tr-$i.json > $O/$tr-$i.log 2>&1
    rc=$?
    echo "$tr-$i rc $rc $(grep -ao '[0-9]* checks, [0-9]* failed' $O/$tr-$i.log | tail -1) FAIL lines $(grep -ac '^FAIL' $O/$tr-$i.log) passes $(grep -acE '^(PASS|FAIL) layer[0-9]+/rows[0-9]+(/rep[0-9]+)?(/pipelined)?/rank0/out_stream' $O/$tr-$i.log) secs $(( $(date +%s) - t0 )) $(date +%T)" >> $O/runs.txt
    grep -a '^FAIL' $O/$tr-$i.log | head -4 >> $O/runs.txt
  done
done
unset MEMRA_TP_TRANSPORT
echo "STRESS2-DONE $(date +%T)" >> $O/runs.txt
