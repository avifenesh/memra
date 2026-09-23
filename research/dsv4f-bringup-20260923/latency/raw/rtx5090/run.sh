#!/bin/bash
# Latency-lane component run on the local 5090: base and lane correctness, then an
# interleaved timing A/B (N=5, both orders) with 250 ms telemetry. Runs under the rig lock.
set -u
J=/home/avifenesh/.claude/jobs/4b9440c5/tmp
O=$J/lat5090
cd $O
export NVIDIA_TF32_OVERRIDE=0
{
  echo "start $(date -Is)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  for t in base lane; do
    $J/lat-$t.bin --ignored --test-threads=1 --nocapture --skip latency_kernel_timing > $O/correct-$t.log 2>&1
    echo "correct $t rc=$?"
  done
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv -lms 250 > $O/telemetry.csv &
  TEL=$!
  for r in 1 2 3 4 5; do
    if [ $((r % 2)) -eq 1 ]; then order="base lane"; else order="lane base"; fi
    for t in $order; do
      DSV4_LATENCY_TREE=$t $J/lat-$t.bin --ignored --test-threads=1 --nocapture --exact latency_kernel_timing > $O/timing-r$r-$t.log 2>&1
      echo "timing r$r $t rc=$?"
    done
  done
  kill $TEL
  echo "done $(date -Is)"
} > $O/run.log 2>&1
