#!/usr/bin/env bash
# GB10 counters run for memra#1311: dsv41-tpbench of each build in TAGS at each device list of DEVS (ranks on the one
# card), LAYERS layers resident, device-push, row-block calls of 4 to 16 rows, 2 rounds. Only the gathers' counters of
# each call are read (collectives, copies, bytes); the times of ranks sharing one card mean nothing.
# usage: tb.sh "TAGS" "DEVS" [LAYERS]   Done-file out/tb/runs.txt, marker TB-DONE.
set -u
TAGS=$1; DEVS=$2; KL=${3:-4}
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel MEMRA_TP_TRANSPORT=device-push
O=$R/out/tb; mkdir -p $O; : > $O/runs.txt
PF=""
for n in 4 8 9 10 11 12 16; do PF="$PF${PF:+,}$R/prompts/rows$n.i32"; done
for dev in $DEVS; do
  for tag in $TAGS; do
    o=$O/$tag-$dev
    rm -rf "$o"
    sha256sum $R/bins/$tag/dsv41-tpbench >> $O/runs.txt
    timeout 8m $R/bins/$tag/dsv41-tpbench $C --token-map $TM --devices $dev --out $o --layers $KL --resident $KL \
      --prefill $PF --rounds 2 > $O/$tag-$dev.log 2>&1
    echo "$tag $dev layers $KL rc $? rows $(grep -c . $o/cells.jsonl 2>/dev/null) $(grep -a -m1 'dsv41-tp\]' $O/$tag-$dev.log | cut -c1-120) $(date +%T)" >> $O/runs.txt
    grep -a -i "error\|refus" $O/$tag-$dev.log | head -3 >> $O/runs.txt
  done
done
echo "TB-DONE $(date +%T)" >> $O/runs.txt
