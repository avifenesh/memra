#!/usr/bin/env bash
# Runs ON the GB10 host: one served arm end to end.
# usage: run_arm.sh OUT BIN "EXTRA env" "targets" "reps" [WRAP]
#   boots serve.sh, waits for /readyz (15 min cap), runs pf_long_memra.py, stops the server.
set -u
OUT=$1; BIN=$2; EXTRA=${3:-}; TARGETS=${4:-"32768 131072 260000"}; REPS=${5:-"3 2 1"}; WRAP=${6:-}
D=$HOME/.cache/memra-attn-gb10
mkdir -p "$OUT"
echo "$(date -Is) run_arm OUT=$OUT BIN=$BIN EXTRA=$EXTRA TARGETS=$TARGETS REPS=$REPS WRAP=$WRAP" | tee "$OUT/run.log"
OUT=$OUT BIN=$BIN EXTRA=$EXTRA WRAP=$WRAP bash "$D/box/serve.sh" &
sp=$!
sleep 2
spid=$(cat "$OUT/server.pid")
if ! waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
     --fail-if "! kill -0 $spid" --every 5s --timeout 15m >> "$OUT/run.log" 2>&1; then
  echo "$(date -Is) server never became ready" | tee -a "$OUT/run.log"
  kill -TERM "$(cat "$OUT/server.pid")" 2>/dev/null; wait $sp; exit 3
fi
echo "$(date -Is) ready" | tee -a "$OUT/run.log"
python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets $TARGETS --reps $REPS --tag "$(basename "$OUT")" >> "$OUT/run.log" 2>&1
echo "$(date -Is) bench rc=$?" | tee -a "$OUT/run.log"
kill -TERM "$(cat "$OUT/server.pid")" 2>/dev/null
waitfor --gone "$(cat "$OUT/server.pid")" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$(cat "$OUT/server.pid")" 2>/dev/null
wait $sp
echo "$(date -Is) done" | tee -a "$OUT/run.log"
