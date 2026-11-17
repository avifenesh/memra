#!/bin/bash
export MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel MEMRA_TP_TRANSPORT=device-push
for b in route2 gather2 main25; do
  for n in 4 9 10; do
    t=/home/access/dsv41/receipts-perf/t1324/$b-rows$n
    mkdir -p $t
    /usr/local/bin/nsys profile -t cuda -s none --cpuctxsw=none -o $t/trace -f true       /home/access/dsv41/perf-bins/$b/dsv41-tpbench /home/access/dsv41/DeepSeek-V4.1-Flash-1017 --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32 --devices 0,1,2,3 --out $t/out       --layers 40 --resident 40 --prefill /home/access/dsv41/receipts-perf/t1324/trows$n.i32 --rounds 6 > $t/run.log 2>&1
    echo "$b-rows$n rc $? $(date +%H:%M:%S)" >> /home/access/dsv41/receipts-perf/t1324/runs.txt
  done
done
