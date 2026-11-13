#!/bin/bash
# Laptop RTX 5090 (inside a gpulock lease): the new build's byte check (small-row body and form 0 against every
# CUTLASS form), then dense cells of the three builds interleaved (k3 old new / new old k3).
set -u
F=$HOME/.cache/dsv41-memra/cutlass-decode/fix
O=$F/out-${1:-chk}-$(date +%m%d-%H%M)
mkdir -p $O
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
timeout 600 $F/bin/probe-new check > $O/check.txt 2>&1; echo "check rc $?" >> $O/runs.txt
i=0
for order in "k3 cutlass new" "new cutlass k3"; do
  for b in $order; do
    i=$((i+1))
    timeout 300 $F/bin/probe-$b ${2:-dense} 64 > $O/run$i-$b.txt 2>&1; echo "run$i $b rc $? $(date +%T)" >> $O/runs.txt
  done
done
kill $SMI
echo "RUN-DONE $O" >> $O/runs.txt
echo $O
