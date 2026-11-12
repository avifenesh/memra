#!/usr/bin/env bash
# GB10 gates lease B (cutlass branch): tp at 2 and 4 ranks on the card, three transports, with the hash log.
# Done-file out/gates-$TAG/b.txt, marker GATES-B2-DONE.
set -u
TAG=f4
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/b2.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/b2.txt; }
for tr in peer-pull host-canonical; do
  for dev in 0,0,0,0; do
    if [ $tr = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$tr; fi
    timeout 13m $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev \
      --decode 2 --token-map $TM --out $O/tp-$tr-$dev.json > $O/tp-$tr-$dev.log 2>&1
    say "tp $dev $tr" $? $O/tp-$tr-$dev.log
    grep -a "^FAIL" $O/tp-$tr-$dev.log | head -5 >> $O/b2.txt
    grep -a "^HASH" $O/tp-$tr-$dev.log | sort | uniq -c | sort -rn | head -4 >> $O/b2.txt
  done
done
unset MEMRA_TP_TRANSPORT
echo "GATES-B2-DONE $(date +%T)" >> $O/b2.txt
