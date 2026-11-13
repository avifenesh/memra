#!/usr/bin/env bash
# memra#1304, runs ON GB10 cb33 inside the caller's sparklock lease: memra#1221's truly-short cell
# (memra#1257's short/ cells, the issue's recipe) for one arm. Refuses to start if anything else
# runs on the unit. A fresh server of the arm's binary (serve_short.sh: Qwen3.8-27B GGUF with its
# MTP draft, 262,144 context, the served defaults, memory guard lowered to 1 GiB), then, with
# TARGET_MB > 0, mem_holder.py takes MemAvailable down to TARGET_MB (256 MB steps; oom_score_adj
# 1000; exits by itself after 300 s), then the requests, then the holder is released and the
# server stops. MODE=single: one g1k request. MODE=pair: a g28k request, then a g1k request once
# the g28k one has started its prime (a second arrival on a busy box). EXTRA adds server env.
# usage: [MODE=single|pair] [EXTRA="K=V ..."] cell.sh CAMPAIGN NN-NAME ARM TARGET_MB
set -u
D=$HOME/.cache/memra-1304; cd "$D" || exit 1
CAMP=$1; NAME=$2; BIN=$D/bins/$3; TGT=$4; MODE=${MODE:-single}
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"; CL=$D/runs/$CAMP/campaign.log
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; docker ps -q 2>/dev/null; ps -eo pid,args | grep -E '[g]inza|[p]ikadon|[m]emra-server')
[ -z "$busy" ] || { echo "$(date -Is) $NAME: unit not idle: $busy" | tee -a "$CL" > "$OUT/run.log"; exit 3; }
t0=$(date '+%F %T')
{ echo "$(date -Is) start, unit idle, arm $3, target ${TGT} MB, mode $MODE, extra ${EXTRA:-}"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } > "$OUT/run.log"
GUARD_KB=1048576 OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1 ${EXTRA:-}" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve_short.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m >> "$OUT/run.log" 2>&1 \
  || { kill -TERM "$spid"; wait $sp; echo "CELL-FAIL $NAME (not ready)" >> "$CL"; exit 1; }
hp=
if [ "$TGT" -gt 0 ]; then
  python3 "$D/box/mem_holder.py" "$TGT" 300 > "$OUT/holder.log" 2>&1 &
  hp=$!
  waitfor --until "awk '/^MemAvailable/{exit !(\$2/1024 < $TGT + 300)}' /proc/meminfo" --fail-if "! kill -0 $hp" --every 2s --timeout 120s >> "$OUT/run.log" 2>&1
fi
{ echo "$(date -Is) before the request:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
if [ "$MODE" = pair ]; then
  n0=$(grep -c 'prime-chunk' "$OUT/server.log")
  python3 "$D/box/idle_req.py" "$OUT/ab.jsonl" "$NAME-a" 28000 >> "$OUT/run.log" 2>&1 &
  rp=$!
  waitfor --until "[ \$(grep -c prime-chunk $OUT/server.log) -gt $n0 ]" --fail-if "! kill -0 $rp" --every 1s --timeout 60s >> "$OUT/run.log" 2>&1
  { echo "$(date -Is) first request priming, second sent:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
  python3 "$D/box/idle_req.py" "$OUT/ab.jsonl" "$NAME-b" 1024 >> "$OUT/run.log" 2>&1
  waitfor --gone "$rp" --timeout 8m >> "$OUT/run.log" 2>&1 || kill -TERM "$rp"
  wait "$rp"
else
  python3 "$D/box/idle_req.py" "$OUT/ab.jsonl" "$NAME" 1024 >> "$OUT/run.log" 2>&1
fi
{ echo "$(date -Is) after the request:"; grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo; } >> "$OUT/run.log"
if [ -n "$hp" ]; then
  kill -TERM "$hp"; waitfor --gone "$hp" --timeout 1m >> "$OUT/run.log" 2>&1 || kill -KILL "$hp"
  wait "$hp"
fi
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
journalctl -k --since "$t0" --no-pager > "$OUT/kernel.log" 2>&1
grep -ciE 'out of memory|oom-kill|oom_reaper|invoked oom-killer' "$OUT/kernel.log" > "$OUT/kernel-oom-count.txt"
grep -E '\[(spec|admission|admit-oom|admit-trim|admit-cal|device-trim|prefix-cache)\]|out of memory|OUT_OF_MEMORY' "$OUT/server.log" | grep -v '\[prefix-cache\] lookup' > "$OUT/trace.txt"
grep '^{' "$OUT/ab.jsonl" | cut -c1-300 | sed "s/^/${NAME} /" >> "$CL"
echo "CELL-DONE $NAME kernel-oom-lines $(cat "$OUT/kernel-oom-count.txt")" >> "$CL"
