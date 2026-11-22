#!/usr/bin/env bash
# GB10 box, step 11: boot one memra-server (BIN) on one artifact directory (MODEL), run the
# readings (bench.py: the long probe at c=1 and c=4) and/or the six chat turns (real_turns.py),
# both sampled at the artifact's generation_config with seed 20261121, and/or the served-answer
# gates (chat_gate.py), then stop it. One server at a time.
# usage: s11-serve.sh TAG BIN MODEL WHAT [env assignments...]   (WHAT: comma list of bench,turns,gate)
set -u
tag=$1; bin=$2; model=$3; what=$4; shift 4
D=$HOME/hy4flash/s11
S=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
out=$D/serve/$tag
mkdir -p "$out"
port=18391
key=hy4f-serve-gate
log=$out/server.log
if ! waitfor --quiet --timeout 30m --every 10s --no-proc 'release/(hy4f-gpu|hy4f-run|hy4f-spec|memra-server)|bins-[a-z0-9]+/(memra-server|hy4f-gpu)' > /dev/null \
   || ! waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null; then
  echo "REFUSED $(date '+%F %T'): another GPU job or too little memory; not starting" > "$out/run.txt"
  exit 2
fi
echo "START $(date '+%F %T') tag=$tag bin=$bin model=$model what=$what env=$*" > "$out/run.txt"
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
# 250 ms GPU telemetry for the run's readings
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader -lms 250 > "$out/telemetry.csv" 2>&1 &
tel=$!
rc=0
has() { [[ ",$what," == *",$1,"* ]]; }
if has gate; then
  python3 $S/chat_gate.py 127.0.0.1:$port $key hy4 $HOME/hy4flash/pr7/requests $D/gate-in/tapes.json \
    $D/gate-in/route-gen.json "$out/chat-gate.json" "$log" ${QUICK:+--quick} > "$out/chat-gate.log" 2>&1
  rc=$?
  echo "GATE rc=$rc $(date '+%F %T')" >> "$out/run.txt"
fi
if has bench; then
  python3 $S/bench.py 127.0.0.1:$port $key hy4 $HOME/hy4flash/pr7/requests "$out/bench.jsonl" \
    --server-log "$log" > "$out/bench.log" 2>&1
  brc=$?
  echo "BENCH rc=$brc $(date '+%F %T')" >> "$out/run.txt"
  [ $brc -ne 0 ] && [ $rc -eq 0 ] && rc=$brc
fi
if has turns; then
  python3 $S/real_turns.py 127.0.0.1:$port $key hy4 "$out/turns.json" ${TURNS_GREEDY:+--greedy} --runs ${TURNS_RUNS:-3} > "$out/turns.log" 2>&1
  trc=$?
  echo "TURNS rc=$trc $(date '+%F %T')" >> "$out/run.txt"
  [ $trc -ne 0 ] && [ $rc -eq 0 ] && rc=$trc
fi
kill $tel 2>/dev/null
grep -c 'hy4f-step' "$log" >> "$out/run.txt"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> "$out/run.txt" 2>&1
kill -TERM $pid
waitfor --quiet --timeout 3m --every 2s --gone $pid > /dev/null || kill -9 $pid
wait $pid 2>/dev/null
echo "STOPPED $(date '+%F %T') rc=$rc" >> "$out/run.txt"
exit $rc
