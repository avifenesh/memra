#!/usr/bin/env bash
# GB10 box, batch lane: boot one memra-server (BIN) on the staged artifact, run conc_bench.py
# (long + chat at the given concurrencies, sampled, seeded), stop it. One server at a time.
# usage: serve.sh TAG BIN [env assignments...]   (CONC, RUNS, WORK env override the bench's)
set -u
tag=$1; bin=$2; shift 2
D=$HOME/hy4flash/batch
S=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
model=$HOME/hy4flash/hf-stage
out=$D/serve/$tag
mkdir -p "$out"
port=18392
key=hy4f-batch
log=$out/server.log
if ! waitfor --quiet --timeout 30m --every 10s --no-proc 'release/(hy4f-gpu|hy4f-run|hy4f-spec|memra-server)|bins-[a-z0-9]+/(memra-server|hy4f-gpu)' > /dev/null \
   || ! waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null; then
  echo "REFUSED $(date '+%F %T'): another GPU job or too little memory" > "$out/run.txt"
  exit 2
fi
echo "START $(date '+%F %T') tag=$tag bin=$bin env=$*" > "$out/run.txt"
sha256sum $bin >> "$out/run.txt"
free -g | head -2 >> "$out/run.txt"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> "$out/run.txt" 2>&1
env "$@" MEMRA_COMPAT=memra MEMRA_MODELS="hy4=$model" MEMRA_ADDR="127.0.0.1:$port" MEMRA_API_KEY=$key \
  systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $bin > "$log" 2>&1 &
pid=$!
echo "server pid $pid" >> "$out/run.txt"
if ! waitfor --quiet --timeout 15m --every 5s --http "http://127.0.0.1:$port/readyz" > /dev/null; then
  echo "NOT READY $(date '+%F %T')" >> "$out/run.txt"
  kill $pid; wait $pid
  tail -n 20 "$log" >> "$out/run.txt"
  exit 1
fi
echo "READY $(date '+%F %T')" >> "$out/run.txt"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader -lms 250 > "$out/telemetry.csv" 2>&1 &
tel=$!
python3 $S/conc_bench.py 127.0.0.1:$port $key hy4 $HOME/hy4flash/pr7/requests "$out/conc.jsonl" \
  --server-log "$log" --conc ${CONC:-1,2,4,8} --runs ${RUNS:-3} --work ${WORK:-long,chat} > "$out/conc.log" 2>&1
rc=$?
echo "CONC rc=$rc $(date '+%F %T')" >> "$out/run.txt"
kill $tel 2>/dev/null
kill -TERM $pid
waitfor --quiet --timeout 3m --every 2s --gone $pid > /dev/null || kill -9 $pid
wait $pid 2>/dev/null
echo "STOPPED $(date '+%F %T') rc=$rc" >> "$out/run.txt"
exit $rc
