#!/usr/bin/env bash
# b23d: verify final + the five red arms. usage: verify-chain.sh PREFIX
set -u
D=$HOME/hy4flash/perf; p=$1
L=$HOME/hy4flash/logs/perf-verify-chain-$p.log
: > $L
for spec in "final" "red-engram engram-off" "red-dequant dequant-bf16" "red-ties ties-reversed" \
            "red-kv kv-off-by-one" "red-pool pool-early"; do
  set -- $spec
  echo "RUN $spec $(date '+%F %T')" >> $L
  bash $D/verify.sh "$p-$1" ${2:-}
  tail -n 3 $HOME/hy4flash/logs/perf-verify-$p-$1.log >> $L
done
echo "VERIFY_CHAIN_DONE $(date "+%F %T")" >> $L
