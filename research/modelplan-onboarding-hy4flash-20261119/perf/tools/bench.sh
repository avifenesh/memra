#!/usr/bin/env bash
# c=1 throughput reading: chat (36) and long (2104) prefill + 64 decode calls sampled as generation_config.json says (seeded), 1 warm-up + N runs
# usage: bench.sh TAG [N]   (BIN= overrides the binary)
set -u
tag=$1; n=${2:-3}
D=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/perf; bin=${BIN:-$D/target/release/hy4f-gpu}
mkdir -p $D/bench
for req in chat long; do
  log=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/logs/perf-bench-$tag-$req.log
  waitfor --quiet --timeout 10m --every 10s --until $D/memwait.sh > /dev/null
  echo "START $(date '+%F %T') tag=$tag req=$req bin=$bin sha=$(sha256sum $bin | cut -c1-16)" > $log
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv >> $log
  systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -- \
    $bin ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/hf-stage --request ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/pr7/requests/$req.json --bench $n > $D/bench/$tag-$req.jsonl 2>> $log
  echo "EXIT rc=$? $(date '+%F %T')" >> $log
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv >> $log
done
echo "BENCH_DONE $tag $(date '+%F %T')" >> ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/logs/perf-bench-$tag-long.log
