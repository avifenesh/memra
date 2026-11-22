#!/usr/bin/env bash
# one hy4f-gpu tape run. usage: run-gpu.sh TAG REQUEST [args]   (BIN= overrides the binary)
set -u
tag=$1 req=$2; shift 2
D=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/perf; bin=${BIN:-$D/target/release/hy4f-gpu}
log=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/logs/perf-gpu-$tag.log
mkdir -p $D/native
waitfor --quiet --timeout 10m --every 10s --until $D/memwait.sh > /dev/null
echo "START $(date "+%F %T") tag=$tag req=$req bin=$bin args=$*" > $log
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $bin ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/hf-stage --request ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/pr7/requests/$req.json --tape $D/native/$tag "$@" >> $log 2>&1
echo "EXIT rc=$? $(date "+%F %T")" >> $log
