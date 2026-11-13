#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1221 part 2): one boot of the 29-turn agentic
# replay (the 2026-11-05 decision cell's workload and driver, research/c8-workspace-20261108/
# harness/agent_turns.py: a prime pass at max_tokens 1, then the timed pass) on the MTP route the
# GB10 launcher serves (serve.sh: Qwen3.8-27B NVFP4 + Q5_K with its masked MTP draft), ctx 32,768,
# 8 sessions, prefix cache 4,000 MB, no host tier, c = 2, thinking on, temperature 0, seed 1234,
# 768 tokens. MEMRA_TICK_TRACE=1 prints the prefix cache's loan reconcile lines. After the replay
# the leak-check probe (tools/trim_probe.py, both arms) drops every evictable pool through the
# stock trim path and prints the pool's used bytes and the cache's charged bytes against the boot
# baseline. usage: agentic_cell.sh CAMPAIGN NN-ARM BIN_DIR   -> runs/CAMPAIGN/NN-ARM/
set -u
D=$HOME/.cache/memra-1221b
CAMP=$1; NAME=$2; BIN=$D/bins/$3
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
[ -x "$BIN/memra-server" ] || { echo "AGENTIC-CELL-FAIL $NAME (no binary)" >> "$D/runs/$CAMP/campaign.log"; exit 1; }
OUT=$OUT BIN=$BIN WRAP="python3 $D/box/tsrun.py" \
  EXTRA="MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_PREFIX_CACHE_MB=4000 MEMRA_TICK_TRACE=1 PROBE_TRIM_FILE=$OUT/trim-now" \
  bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
rc=0
if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1; then
  CLIENT_MAX_TIME=900 BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 timeout 1500 \
    python3 "$D/box/agent_turns.py" "$D/box/ab-workload.json" "$OUT" "$NAME" --users 2 --think 1 \
    --temperature 0 --seed 1234 --max-tokens 768 >> "$OUT/cells.log" 2>&1 || rc=$?
  curl -s http://127.0.0.1:8090/metrics > "$OUT/metrics.json" 2>/dev/null
  touch "$OUT/trim-now"
  # The worker checks the file at a tick top; an idle worker waits for a command, so a
  # one-token request wakes it (the trim runs at the top of the tick that admits it).
  curl -s -m 120 -o /dev/null http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"q38","prompt":"wake","max_tokens":1,"temperature":0}'
  waitfor --grep '\[probe-trim\] after trim' "$OUT/server.log" --timeout 3m >> "$OUT/run.log" 2>&1 || rc=4
else
  echo "boot FAILED" >> "$OUT/run.log"; rc=3
fi
kill -TERM "$spid" 2>/dev/null
waitfor --gone "$spid" --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
{ for pat in 'rewind refused below a lent prefix row' '[spec] rewind below a lent prefix row' '[prefix-loan] reconcile' '[prefix-loan] compact' \
             'insert (spec-boundary)' '[prefix-cache] hit' 'spec restore' 'out of memory' '[probe-trim]'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server.log")"
  done; grep -a -F '[probe-trim]' "$OUT/server.log" | cut -c1-240; } > "$OUT/summary.txt"
tail -n 1 "$OUT/cells.log" | cut -c1-400 | sed "s/^/$NAME /" >> "$D/runs/$CAMP/campaign.log"
gzip -f "$OUT/server.log"
[ $rc = 0 ] && echo "AGENTIC-CELL-DONE $NAME" >> "$D/runs/$CAMP/campaign.log"
exit $rc
