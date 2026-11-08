#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: boot BIN with every server log line timestamped
# (tsrun.py, MEMRA_TICK_TRACE=1), one cold request per length, stop. usage: ts_probe.sh OUT BIN
set -u
OUT=$1; BIN=$2
D=$HOME/.cache/memra-gb10-r4
mkdir -p "$OUT"
OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
sp=$!
sleep 2
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; exit 3; }
echo "$(date -Is) ready" >> "$OUT/run.log"
python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets 32768 131072 260000 --reps 1 1 1 --tag ts-probe >> "$OUT/run.log" 2>&1
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
echo TS-PROBE-DONE >> "$OUT/run.log"
