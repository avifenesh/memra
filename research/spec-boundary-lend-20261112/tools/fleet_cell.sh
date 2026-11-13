#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1221 part 2): one boot of the agent-shaped
# fleet replay (box/fleet_replay_t0.py: tools/fleet-replay.py made greedy, its ok lines carrying the
# completion's sha256) on the GB10 launcher's MTP route (serve.sh) with the prefix cache on, ctx
# 32,768, 8 sessions: 12 conversations in 6 pairs that share a 3-4k-token system prefix under 4
# tenant salts, 2-4-turn bursts, requests one at a time, seed 1234, 80 requests. Entries are lent
# by live sessions, read while their lenders are parked, adopted when the pool evicts a lender,
# and evicted under the cache budget. Then the leak-check trim (both arms carry trim_probe.py).
# usage: fleet_cell.sh CAMPAIGN NN-ARM BIN_DIR   -> runs/CAMPAIGN/NN-ARM/
set -u
D=$HOME/.cache/memra-1221b
CAMP=$1; NAME=$2; BIN=$D/bins/$3
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
[ -x "$BIN/memra-server" ] || { echo "FLEET-CELL-FAIL $NAME (no binary)" >> "$D/runs/$CAMP/campaign.log"; exit 1; }
OUT=$OUT BIN=$BIN WRAP="python3 $D/box/tsrun.py" \
  EXTRA="MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_PREFIX_CACHE_MB=4000 MEMRA_TICK_TRACE=1 PROBE_TRIM_FILE=$OUT/trim-now" \
  bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
rc=0
if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1; then
  timeout 1500 python3 "$D/box/fleet_replay_t0.py" --base http://127.0.0.1:8090 --model q38 \
    --duration 1400 --requests-per-minute 600 --requests 80 --seed 1234 > "$OUT/replay.json" 2> "$OUT/replay.log" || rc=$?
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
             'insert (spec-boundary)' '[prefix-cache] hit' 'spec restore' 'evict (LRU)' 'spec-park' 'out of memory' '[probe-trim]'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server.log")"
  done; grep -a -F '[probe-trim]' "$OUT/server.log" | cut -c1-240; } > "$OUT/summary.txt"
echo "$NAME $(tail -c 400 "$OUT/replay.json" | tr -d '\n' | cut -c1-300)" >> "$D/runs/$CAMP/campaign.log"
gzip -f "$OUT/server.log"
[ $rc = 0 ] && echo "FLEET-CELL-DONE $NAME" >> "$D/runs/$CAMP/campaign.log"
exit $rc
