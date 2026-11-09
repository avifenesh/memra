#!/bin/bash
# deepseek-v4.1 serving gates on one card, one lease each, every server stopped inside its lease:
#   ckpt    `memra model verify checkpoint` with the native side on the residency route
#   verify  `memra model verify serve` on the pinned checkpoint, from that CheckpointParity bundle
#   chat    a memra-server of our own: render, solo/batched/streamed/sampled answers against the native
#           greedy runs, the numeric budget, one longer generation for the decode rate
#   bench   the same server driven by the lane's shared benchmark client (token-id completions, streamed, usage)
# usage: serve_gate.sh [ckpt|verify|chat|all|bench]
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
D=$ROOT
S=$D/serve
SRC=$D/src-serve
T=$D/target-serve/release
TM=$D/oracle/engram_constants.token_map.i32
UUID=$GPU_UUID
RCP=$SRC/research/modelplan-onboarding-dsv41-20261109/gpu/receipts/checkpoint/verify-capital-rel
what=${1:-all}
mkdir -p $S/logs
sha256sum $T/memra-server $T/memra
grep -h MemAvailable /proc/meminfo
lease() {
  local name=$1 to=$2; shift 2
  if [ -e $QUIET_FILE ]; then echo "QUIET is set: not starting $name"; exit 3; fi
  if [ -e $ROOT/serve/STOP ]; then echo "STOP is set: not starting $name"; exit 4; fi
  local R=$D/leases/serve-$name-$(date +%Y%m%dT%H%M%S); mkdir -p "$R"
  echo "lease $name -> $R"
  (cd $D && $GPU_RUN --gpus $UUID --receipt "$R" --box-lock $BOX_LOCK --timeout $to \
    --wait 1800 -- taskset -c $CORES "$@")
  echo "$name rc $?"
}
OUT=$S/verify-res
if [ "$what" = ckpt ] || [ "$what" = all ]; then
  # CheckpointParity again, the native side on the residency route (the wrapper adds --experts-host)
  rm -rf $OUT; mkdir -p $OUT
  gunzip -c $RCP/vendor-cuda-capital.tsv.gz > $S/vendor-cuda-capital.tsv
  sha256sum $T/dsv41-run $S/scripts/dsv41-run-experts-host
  lease ckpt 2400 env DSV41_RUN=$T/dsv41-run MEMRA_DSV41_TOKEN_MAP=$TM \
    $T/memra model verify checkpoint DeepSeek-V4.1-Flash-1017 --against deepseek_v41 --oracle $S/vendor-cuda-capital.tsv \
    --native-runner $S/scripts/dsv41-run-experts-host --out $OUT > $S/logs/verify-checkpoint-res.log 2>&1
  tail -n 3 $S/logs/verify-checkpoint-res.log
  cat $OUT/checkpoint-parity.tsv $OUT/gates.txt 2>/dev/null
fi
if [ "$what" = verify ] || [ "$what" = all ]; then
  if [ ! -f $OUT/checkpoint-parity.tsv ]; then
    rm -rf $OUT; mkdir -p $OUT
    cp $RCP/artifact.lock $RCP/checkpoint-parity.tsv $OUT/
  fi
  lease verify 2400 env MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_SERVE_VERIFY_TIMEOUT_S=1500 \
    $T/memra model verify serve DeepSeek-V4.1-Flash-1017 --against deepseek_v41 --out $OUT --native-runner $T/memra-server \
    > $S/logs/verify-serve.log 2>&1
  tail -n 3 $S/logs/verify-serve.log
  cat $OUT/serve-gate.tsv $OUT/gates.txt 2>/dev/null
fi
if [ "$what" = chat ] || [ "$what" = all ]; then
  lease chat 3600 bash $S/scripts/chat_run.sh
  tail -n 15 $S/logs/chat-gate.log
fi
if [ "$what" = bench ]; then
  # the lane's shared benchmark client against the route (wiring, two samples a workload); not part of `all`
  lease bench 3600 bash $S/scripts/bench_run.sh
fi
echo SERVE-GATE-DONE
