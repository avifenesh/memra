#!/bin/bash
# Laptop RTX 5090 (inside a gpulock lease): the timing proxy. The new build's byte checks (dense forms, the MoE path
# against the old path), then every cell (dense, dense set, MoE) of the three builds interleaved, ROUNDS rounds of
# k3 old new / new old k3, SM clock and power sampled every 250 ms.
set -u
F=$HOME/.cache/dsv41-memra/cutlass-decode/fix
B=${BIN:-$F/bin}
O=$F/out-${1:-full}-$(date +%m%d-%H%M)
ROUNDS=${2:-2}
mkdir -p $O
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
sha256sum $B/probe-* > $O/bins.sha256
timeout 600 $B/probe-new check > $O/check.txt 2>&1; echo "check rc $? $(grep '# check' $O/check.txt)" >> $O/runs.txt
i=0
for r in $(seq 1 $ROUNDS); do
  for order in "k3 cutlass new" "new cutlass k3"; do
    for b in $order; do
      i=$((i+1))
      echo "run$i $b start $(date +%T)" >> $O/runs.txt
      timeout 900 $B/probe-$b all 64 > $O/run$i-$b.txt 2>&1; echo "run$i $b rc $? end $(date +%T)" >> $O/runs.txt
    done
  done
done
kill $SMI
echo "RUN-DONE $O" >> $O/runs.txt
echo $O
