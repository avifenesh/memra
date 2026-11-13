#!/usr/bin/env bash
# GB10 gates C at #1286's restacked head, on the DEFAULT dense form (no --dense-form), every line labelled with it:
# decode on three captures, launch by launch, and the decode graph door at one rank and two ranks (device-push).
# usage: gc.sh TAG   Done-file out/gates-TAG/c.txt, marker GATES-C-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/c.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/c.txt; }
sha256sum $B/dsv41-gate >> $O/c.txt
for cap in dec/captures/straddle126 dec/captures/gsm captures-v2/capital; do
  name=${cap##*/}
  case $name in capital) L=1,2,3,14,20,21,24 ;; *) L=0,1,2,3,8,14,20,21,24 ;; esac
  timeout 7m $B/dsv41-gate decode $C $D/$cap --layers $L --token-map $TM --out $O/decode-$name.json > $O/decode-$name.log 2>&1
  say "decode $name, dense form default" $? $O/decode-$name.log
done
grep -a "^FAIL" $O/decode-*.log | head -8 >> $O/c.txt
export MEMRA_DSV41_DECODE_GRAPH=1
for dev in 0 0,0; do
  if [ $dev = 0 ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=device-push; fi
  timeout 7m $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev \
    --decode 2 --token-map $TM --out $O/tp-graph-$dev.json > $O/tp-graph-$dev.log 2>&1
  say "tp graph $dev, dense form default" $? $O/tp-graph-$dev.log
  grep -a "^FAIL" $O/tp-graph-$dev.log | head -6 >> $O/c.txt
done
unset MEMRA_TP_TRANSPORT MEMRA_DSV41_DECODE_GRAPH
echo "GATES-C-DONE $(date +%T)" >> $O/c.txt
