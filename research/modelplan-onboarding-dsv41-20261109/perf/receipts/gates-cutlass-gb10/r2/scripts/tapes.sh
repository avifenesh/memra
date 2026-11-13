#!/usr/bin/env bash
# The order-gate tapes of one group (G = chat | long) at #1286's restacked head, teacher-forced through the whole trunk
# with the route log, every file compared byte for byte with the CUTLASS build's tapes (ref/tapes-ref-G.sha256 =
# receipts gates-cutlass-gb10/f4/tapes-f4-G.sha256; paths compared without their leading "./" or "G/").
# usage: tapes.sh TAG G   Done-file out/tapes-TAG/G-runs.txt, marker TAPES-G-DONE.
set -u
TAG=$1; G=$2
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/tapes-$TAG; mkdir -p $O/$G
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
norm() { awk -v g="$G/" '{p=$2; sub(/^\.\//, "", p); if (index(p, g) == 1) p = substr(p, length(g) + 1); print $1, p}' "$1" | sort -k2; }
sed "s|/home/access/dsv41/dec/ids|$D/dec/ids|g" $D/dec/$G-forced.txt > $O/$G-forced.txt
echo "$G start $(date +%T), $(avail) GB available, dense form default, $(sha256sum $B/dsv41-run | cut -c1-16)" > $O/$G-runs.txt
timeout 29m $B/dsv41-run $C --token-map $TM --cases $O/$G-forced.txt --out $O/$G --routes > $O/$G.log 2>&1
echo "$G rc $? $(grep -h 'dsv41-run: [0-9]* forwards\|route log' $O/$G.log | tail -n 2 | tr '\n' ' ') end $(date +%T), $(avail) GB available" >> $O/$G-runs.txt
(cd $O/$G && find . -type f | sort | xargs sha256sum) > $O/$G.sha256
REF=$R/ref/tapes-ref-$G.sha256
n=$(wc -l < $O/$G.sha256)
bad=$(diff <(norm $O/$G.sha256) <(norm $REF) | grep -c '^[<>]')
echo "$G tapes: $n files (the route log and meta.json included), ref $(wc -l < $REF) files, $bad diff lines against the CUTLASS set ($([ $bad = 0 ] && echo BYTE-IDENTICAL || echo DIFFER))" >> $O/$G-runs.txt
echo "TAPES-$G-DONE $(date +%T)" >> $O/$G-runs.txt
