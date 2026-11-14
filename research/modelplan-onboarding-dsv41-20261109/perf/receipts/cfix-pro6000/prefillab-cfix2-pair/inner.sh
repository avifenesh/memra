#!/bin/bash
set -u
export MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel
if [ default = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=default; fi
i=0
for a in   A:/home/access/dsv41/perf-bins/route2 B:/home/access/dsv41/perf-bins/cfix2 B:/home/access/dsv41/perf-bins/cfix2 A:/home/access/dsv41/perf-bins/route2   A:/home/access/dsv41/perf-bins/route2 B:/home/access/dsv41/perf-bins/cfix2 B:/home/access/dsv41/perf-bins/cfix2 A:/home/access/dsv41/perf-bins/route2 ; do
  i=$((i+1)); arm=${a%%:*}; BD=${a#*:}
  rm -rf /home/access/dsv41/receipts-perf/prefillab-cfix2-pair/run$i-$arm
  bash /home/access/dsv41/bin/perf_slow.sh run /home/access/dsv41/receipts-perf/prefillab-cfix2-pair/slowdown.txt run$i-$arm $BD/dsv41-tpbench /home/access/dsv41/DeepSeek-V4.1-Flash-1017 --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32     --out /home/access/dsv41/receipts-perf/prefillab-cfix2-pair/run$i-$arm --devices 0,1 --layers 20 --resident 20     --prefill /home/access/dsv41/bench/prompts/prefill-4096.i32,/home/access/dsv41/bench/prompts/prefill-32768.i32,/home/access/dsv41/bench/prompts/prefill-131072.i32 --rounds 3 > /home/access/dsv41/receipts-perf/prefillab-cfix2-pair/run$i-$arm.log 2>&1
  echo "run$i-$arm rc $?" >> /home/access/dsv41/receipts-perf/prefillab-cfix2-pair/runs.txt
done
