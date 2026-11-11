#!/usr/bin/env bash
# bs1 decode gates on three captures (one lease, <= 30 min)
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/bs1
O=$R/out/dec-bs1; mkdir -p $O; : > $O/gates.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/gates.txt; }
$B/dsv41-gate decode $C $D/dec/captures/straddle126 --layers 0,1,2,3,8,14,20,21,24 --token-map $TM --out $O/decode-straddle126.json > $O/decode-straddle126.log 2>&1
say "decode straddle126" $? $O/decode-straddle126.log
$B/dsv41-gate decode $C $D/dec/captures/gsm --layers 0,1,2,3,8,14,20,21,24 --token-map $TM --out $O/decode-gsm.json > $O/decode-gsm.log 2>&1
say "decode gsm" $? $O/decode-gsm.log
$B/dsv41-gate decode $C $D/captures-v2/capital --layers 1,2,3,14,20,21,24 --token-map $TM --out $O/decode-capital.json > $O/decode-capital.log 2>&1
say "decode capital" $? $O/decode-capital.log
echo "DEC-BS1-DONE $(date +%T)" >> $O/gates.txt
