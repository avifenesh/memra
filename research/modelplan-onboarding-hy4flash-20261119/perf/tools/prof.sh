#!/usr/bin/env bash
# b23d: nsys profile of hy4f-gpu greedy gen. usage: prof.sh TAG REQ GEN [BIN]
set -u
tag=$1 req=$2 gen=$3; bin=${4:-$HOME/hy4flash/perf/target/release/hy4f-gpu}
P=$HOME/hy4flash/perf/prof; mkdir -p $P
log=$HOME/hy4flash/logs/perf-prof-$tag.log
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/perf/memwait.sh > /dev/null
echo "START $(date '+%F %T') tag=$tag req=$req gen=$gen bin=$bin" > $log
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -- \
  nsys profile -t cuda,osrt --cuda-memory-usage=false -s none --cpuctxsw=none -o $P/$tag -f true \
  $bin $HOME/hy4flash/hf-stage --request $HOME/hy4flash/pr7/requests/$req.json --gen $gen >> $log 2>&1
echo "EXIT rc=$? $(date '+%F %T')" >> $log
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f csv -o $P/$tag $P/$tag.nsys-rep >> $log 2>&1
nsys stats -r cuda_gpu_trace -f csv -o $P/$tag $P/$tag.nsys-rep >> $log 2>&1
echo "STATS_DONE $(date '+%F %T')" >> $log
