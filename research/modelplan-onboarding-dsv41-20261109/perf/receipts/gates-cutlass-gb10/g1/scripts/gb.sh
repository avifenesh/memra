#!/usr/bin/env bash
# GB10 gates B at #1286's restacked head: tp on the card at the given device lists and transports, with the hash log,
# on the DEFAULT dense form (the shipping one: no --dense-form), every line labelled with it.
# usage: gb.sh TAG PART "DEVLISTS" "TRANSPORTS"   Done-file out/gates-TAG/bPART.txt, marker GATES-BPART-DONE.
set -u
TAG=$1; P=$2; DEVS=$3; TRS=$4
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/b$P.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/b$P.txt; }
sha256sum $B/dsv41-gate >> $O/b$P.txt
for tr in $TRS; do
  for dev in $DEVS; do
    if [ $tr = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$tr; fi
    timeout 13m $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev \
      --decode 2 --token-map $TM --out $O/tp-$tr-$dev.json > $O/tp-$tr-$dev.log 2>&1
    say "tp $dev $tr, dense form default" $? $O/tp-$tr-$dev.log
    grep -a "^FAIL" $O/tp-$tr-$dev.log | head -5 >> $O/b$P.txt
    grep -a "^HASH" $O/tp-$tr-$dev.log | sort | uniq -c | sort -rn | head -4 >> $O/b$P.txt
  done
done
unset MEMRA_TP_TRANSPORT
echo "GATES-B$P-DONE $(date +%T)" >> $O/b$P.txt
