#!/usr/bin/env bash
# GB10 kernel gate alone at TAG's binaries (the fold of memra#1327's review changes no kernel; the gate binary is new).
# usage: gk.sh TAG   Done-file out/gates-TAG/k.txt, marker GATES-K-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/k.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/k.txt 2>&1
sha256sum $B/dsv41-gate >> $O/k.txt
timeout 20m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
echo "kernels (every form, by construction) rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels.log | tail -1) $(grep -ac '^FAIL' $O/kernels.log) FAIL lines" >> $O/k.txt
grep -a "^FAIL" $O/kernels.log | head -30 >> $O/k.txt
echo "GATES-K-DONE $(date +%T)" >> $O/k.txt
