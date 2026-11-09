#!/bin/bash
# Inside a one-card lease: start memra-server on the deepseek-v4.1 checkpoint and drive it with the lane's shared
# benchmark client (the one the other engines are measured with): token-id prompts on /v1/completions, streamed
# with usage. A wiring check of the client against the route, not a benchmark: two samples per workload.
# usage: bench_run.sh [SUITE] [SAMPLES]      default: smoke,chat_c1,decode_c1,prefill4k 2
# COMPAT=native serves memra's own stream shape (the client reads both); the default is the OpenAI shape.
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
D=$ROOT
S=$D/serve
T=$D/target-serve/release
TM=$D/oracle/engram_constants.token_map.i32
ADDR=127.0.0.1:18342
SUITE=${1:-smoke,chat_c1,decode_c1,prefill4k}
SAMPLES=${2:-2}
OUT=$S/bench-client
rm -rf $OUT
mkdir -p $OUT $S/logs
cd $D
sha256sum $T/memra-server $D/bench/harness/bench.py $D/bench/data/workloads.json
IDX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$GPU_UUID" '$2==u{print $1}')
nvidia-smi --query-gpu=index,power.limit,temperature.gpu,memory.used --format=csv,noheader -i "$IDX"
MEMRA_MODELS=v41=DeepSeek-V4.1-Flash-1017 MEMRA_ADDR=$ADDR MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_COMPAT=${COMPAT:-openai} \
  $T/memra-server > $S/logs/serve-bench.log 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server exited before readiness"; break; fi
  if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
  sleep 2
done
rc=1
if [ $ready = 1 ]; then
  echo "server ready after $SECONDS s"
  python3 $D/bench/harness/bench.py --engine memra --url http://$ADDR --workloads $D/bench/data/workloads.json \
    --out $OUT --label memra-1card-wiring-${COMPAT:-openai} --suite "$SUITE" --samples "$SAMPLES" --gpus "$IDX" --no-ignore-eos \
    --deadline-s 1500 --timeout-s 900 > $S/logs/bench-client.log 2>&1
  rc=$?
fi
kill -TERM $SRV 2>/dev/null
for _ in $(seq 1 120); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -KILL $SRV 2>/dev/null
wait $SRV 2>/dev/null
tail -n 30 $S/logs/bench-client.log
echo "bench client rc $rc"
exit $rc
