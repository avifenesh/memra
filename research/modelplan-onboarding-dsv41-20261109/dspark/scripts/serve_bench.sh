#!/bin/bash
# Inside a lease: start memra-server (target-spec) on the deepseek-v4.1 checkpoint with the arm's environment, drive it
# with the lane's shared benchmark client, stop it and wait for it to leave.
# usage: serve_bench.sh OUT_NAME SUITE SAMPLES [ENV=VALUE ...]      e.g. serve_bench.sh ab/r1-off decode_c1,conc8 1 MEMRA_DSV41_DRAFT=0
# DEVICES (default 0) names the ranks' devices inside the lease; EXTRA_BENCH adds bench.py arguments.
set -u
OUT=$1; SUITE=$2; SAMPLES=$3; shift 3
D=${LANE:?LANE names the working directory on the run box (checkpoint, oracle, dec ids, spec/, target-spec/)}
S=$D/spec
T=$D/target-spec/release
O=$S/out/$OUT
PORT=${PORT:-18361}
ADDR=127.0.0.1:$PORT
mkdir -p $O
cd $D
sha256sum $T/memra-server $D/bench/harness/bench.py $D/bench/data/workloads.json > $O/sha256.txt
nvidia-smi --query-gpu=index,uuid,power.limit,temperature.gpu,fan.speed,memory.used,clocks.sm --format=csv > $O/cards-before.csv
echo "host memory before: $(free -g | sed -n 2p)" > $O/host.txt
printf '%s\n' "$@" > $O/arm.env
IDX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="${CUDA_VISIBLE_DEVICES%%,*}" '$2==u{print $1}')
env MEMRA_MODELS=v41=DeepSeek-V4.1-Flash-1017 MEMRA_ADDR=$ADDR MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
  MEMRA_COMPAT=openai MEMRA_DSV41_DEVICES=${DEVICES:-0} MEMRA_MAX_SESSIONS=${SESSIONS:-8} "$@" \
  $T/memra-server > $O/server.log 2>&1 &
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
    --out $O/bench --label "$OUT" --suite "$SUITE" --samples "$SAMPLES" --gpus "${IDX:-3}" \
    --deadline-s ${DEADLINE:-3000} --timeout-s 1500 ${EXTRA_BENCH:-} > $O/bench.log 2>&1
  rc=$?
fi
kill -TERM $SRV 2>/dev/null
for _ in $(seq 1 120); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -KILL $SRV 2>/dev/null
wait $SRV 2>/dev/null
grep -E "dsv41-serve|dsv41-draft" $O/server.log | head -n 12 | cut -c1-300
tail -n 12 $O/bench.log | cut -c1-400
echo "serve_bench $OUT rc $rc"
exit $rc
