#!/usr/bin/env bash
# Runs ON u1 under the caller's lease: memra#1221's truly-short cell. SIZE=28000: the g28k
# request is sent first and the holder starts once its prime has begun, so the memory goes short
# after admission and before the request's draft-graph capture (at a short arrival admission
# refuses the request outright and the capture guard is never reached). Refuses to start if anything
# else runs on the unit (GPU processes, containers, ginza/pikadon). A fresh server of the arm's
# binary (serve_short.sh, memory guard lowered to GUARD_KB=1 GiB for this cell only), then
# mem_holder.py takes MemAvailable down to TARGET_MB (default 2000; oom_score_adj 1000, exits by
# itself after 300 s), then one g1k request, then the holder is released, then the server stops.
# The kernel log of the window goes to the cell (OOM lines, if any).
# usage: [SIZE=28000] short_cell.sh CAMPAIGN NN-ARM BIN_DIR [TARGET_MB]
set -u
D=$HOME/.cache/memra-1221; cd "$D" || exit 1
CAMP=$1; NAME=$2; BIN=$D/bins/$3; TGT=${4:-2000}
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"; CL=$D/runs/$CAMP/campaign.log
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; docker ps -q 2>/dev/null; ps -eo pid,args | grep -E '[g]inza|[p]ikadon' )
[ -z "$busy" ] || { echo "$(date -Is) $NAME: unit not idle: $busy" | tee -a "$CL" > "$OUT/run.log"; exit 3; }
t0=$(date '+%F %T')
{ echo "$(date -Is) start, unit idle"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > "$OUT/run.log"
GUARD_KB=1048576 OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve_short.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m >> "$OUT/run.log" 2>&1 \
  || { kill -TERM "$spid"; wait $sp; echo "SHORT-CELL-FAIL $NAME (not ready)" >> "$CL"; exit 1; }
if [ "${SIZE:-1024}" = 1024 ]; then
  python3 "$D/box/mem_holder.py" "$TGT" 300 > "$OUT/holder.log" 2>&1 &
  hp=$!
  waitfor --until "awk '/^MemAvailable/{exit !(\$2/1024 < $TGT + 300)}' /proc/meminfo" --fail-if "! kill -0 $hp" --every 2s --timeout 120s >> "$OUT/run.log" 2>&1
  { echo "$(date -Is) before the request:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
  python3 "$D/box/short_req.py" "$OUT/ab.jsonl" "$NAME" >> "$OUT/run.log" 2>&1
else
  n0=$(grep -c 'prime-chunk' "$OUT/server.log")
  python3 "$D/box/short_req.py" "$OUT/ab.jsonl" "$NAME" "$SIZE" >> "$OUT/run.log" 2>&1 &
  rp=$!
  waitfor --until "[ \$(grep -c prime-chunk $OUT/server.log) -gt $n0 ]" --every 1s --timeout 60s >> "$OUT/run.log" 2>&1
  { echo "$(date -Is) prime started, holder on:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
  python3 "$D/box/mem_holder.py" "$TGT" 300 > "$OUT/holder.log" 2>&1 &
  hp=$!
  waitfor --gone "$rp" --timeout 8m >> "$OUT/run.log" 2>&1 || kill -TERM "$rp"
  wait "$rp"
fi
{ echo "$(date -Is) after the request:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
kill -TERM "$hp"; waitfor --gone "$hp" --timeout 1m >> "$OUT/run.log" 2>&1 || kill -KILL "$hp"
wait $hp
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
journalctl -k --since "$t0" --no-pager > "$OUT/kernel.log" 2>&1
grep -ciE 'out of memory|oom-kill|oom_reaper|invoked oom-killer' "$OUT/kernel.log" > "$OUT/kernel-oom-count.txt"
grep -E '\[(spec|admit-oom|admit-trim|device-trim)\]|out of memory|OUT_OF_MEMORY' "$OUT/server.log" > "$OUT/trace.txt"
grep '^{' "$OUT/ab.jsonl" | cut -c1-260 | sed "s/^/${NAME#*-} /" >> "$CL"
grep -q '"error": null' "$OUT/ab.jsonl" || { echo "SHORT-CELL-FAIL $NAME (request)" >> "$CL"; exit 1; }
echo "SHORT-CELL-DONE $NAME kernel-oom-lines $(cat "$OUT/kernel-oom-count.txt")" >> "$CL"
