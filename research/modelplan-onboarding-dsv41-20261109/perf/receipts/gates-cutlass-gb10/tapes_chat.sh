#!/usr/bin/env bash
# cutlass-branch order-gate tapes of ONE group (G = chat | long, from the file name tapes_G.sh): teacher-forced
# through the whole trunk with the route log, one lease of at most 30 min (dsv41-run under `timeout 29m`).
set -u
G=$(basename $0 .sh); G=${G#tapes_}
TAG=$(cat $HOME/.cache/dsv41-cutlass/build-tag)
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data; [ -d $D ] || D=$HOME/.cache/dsv41-cutlass/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/tapes-$TAG; mkdir -p $O/$G
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
sed "s|/home/access/dsv41/dec/ids|$D/dec/ids|g" $D/dec/$G-forced.txt > $O/$G-forced.txt
echo "$G start $(date +%T), $(avail) GB available" > $O/$G-runs.txt
timeout 29m $B/dsv41-run $C --token-map $TM --cases $O/$G-forced.txt --out $O/$G --routes > $O/$G.log 2>&1
echo "$G rc $? $(grep -h 'dsv41-run: [0-9]* forwards\|route log' $O/$G.log | tail -n 2 | tr '\n' ' ') end $(date +%T), $(avail) GB available" >> $O/$G-runs.txt
echo "TAPES-$G-DONE $(date +%T)" >> $O/$G-runs.txt
