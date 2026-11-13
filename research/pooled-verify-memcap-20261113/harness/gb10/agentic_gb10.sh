#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1266): one boot of the 29-turn agentic
# replay (the 2026-11-05 decision cell's workload and driver: a prime pass at max_tokens 1, then
# the timed pass) on the MTP route the GB10 launcher serves (serve.sh: Qwen3.8-27B NVFP4 + Q5_K
# with its masked MTP draft), ctx 32,768, 8 sessions, prefix cache 4,000 MB, no host tier,
# MEMRA_PREFIX_STABLE_BOUNDARY=1, c = USERS (default 2), thinking on, temperature 0, seed 1234,
# 768 tokens. usage: [USERS=N] agentic_gb10.sh CAMPAIGN NAME ARM   -> runs/CAMPAIGN/NAME/
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$D/bins/$3
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
[ -x "$BIN/memra-server" ] || { echo "AGENTIC-CELL-FAIL $NAME (no binary)" >> "$D/runs/$CAMP/campaign.log"; exit 1; }
OUT=$OUT BIN=$BIN WRAP="python3 $D/box/tsrun.py" \
  EXTRA="MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_PREFIX_CACHE_MB=4000 MEMRA_TICK_TRACE=1" \
  bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
rc=0
if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1; then
  CLIENT_MAX_TIME=900 BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 timeout 1500 \
    python3 "$D/box/agent_turns.py" "$D/box/ab-workload.json" "$OUT" "$NAME" --users "${USERS:-2}" --think 1 \
    --temperature 0 --seed 1234 --max-tokens 768 >> "$OUT/cells.log" 2>&1 || rc=$?
  curl -s http://127.0.0.1:8090/metrics > "$OUT/metrics.json" 2>/dev/null
else
  echo "boot FAILED" >> "$OUT/run.log"; rc=3
fi
kill -TERM "$spid" 2>/dev/null
waitfor --gone "$spid" --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
{ for pat in 'out of memory' 'NOT parked' 'graph replay suspended' 'pooled MTP verify capped' 'pre-cohort reclaim' \
             '[prefix-cache] hit' 'spec restore'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server.log")"
  done; } > "$OUT/summary.txt"
tail -n 1 "$OUT/cells.log" | cut -c1-400 | sed "s/^/$NAME /" >> "$D/runs/$CAMP/campaign.log"
gzip -f "$OUT/server.log"
[ $rc = 0 ] && echo "AGENTIC-CELL-DONE $NAME" >> "$D/runs/$CAMP/campaign.log"
exit $rc
