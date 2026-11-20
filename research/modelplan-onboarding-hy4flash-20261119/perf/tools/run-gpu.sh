#!/usr/bin/env bash
# b23d: one hy4f-gpu tape run. usage: run-gpu.sh TAG REQUEST [args]   (BIN= overrides the binary)
set -u
tag=$1 req=$2; shift 2
D=$HOME/hy4flash/perf; bin=${BIN:-$D/target/release/hy4f-gpu}
log=$HOME/hy4flash/logs/perf-gpu-$tag.log
mkdir -p $D/native
waitfor --quiet --timeout 10m --every 10s --until $D/memwait.sh > /dev/null
echo "START $(date "+%F %T") tag=$tag req=$req bin=$bin args=$*" > $log
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $bin $HOME/hy4flash/hf-stage --request $HOME/hy4flash/pr7/requests/$req.json --tape $D/native/$tag "$@" >> $log 2>&1
echo "EXIT rc=$? $(date "+%F %T")" >> $log
