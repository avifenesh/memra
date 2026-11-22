#!/usr/bin/env bash
# memra model verify checkpoint --against hy_v4_flash over the PR 7 f64 oracle tapes,
# the GPU route as the native runner. usage: verify.sh OUT_TAG [RED_ARM]   (BIN= runner)
set -u
D=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/perf
tag=$1; red=${2:-}; runner=${BIN:-$D/target/release/hy4f-gpu}
out=$D/verify/$tag
mkdir -p $D/verify
rm -rf "$out"
log=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/logs/perf-verify-$tag.log
if [ -n "$red" ]; then export HY4F_RUN=$runner HY4F_RED=$red; runner=$D/red-runner.sh; fi
waitfor --quiet --timeout 10m --every 10s --until $D/memwait.sh > /dev/null
echo "START $(date '+%F %T') tag=$tag red=$red runner=$runner sha=$(sha256sum ${HY4F_RUN:-$runner} | cut -c1-16)" > $log
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra model verify checkpoint ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/hf-stage --against hy_v4_flash \
  --out "$out" --oracle ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/pr7/oracle-set --native-runner "$runner" >> $log 2>&1
echo "EXIT rc=$? $(date '+%F %T')" >> $log
