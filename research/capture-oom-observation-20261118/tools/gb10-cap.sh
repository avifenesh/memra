#!/usr/bin/env bash
# Runs ON unit-a inside this lane's lease (memra#1319 / PR #1347's own GB10 cell): one boot of
# arm ARM in memra#1307's MTP serving (serve1307.sh MODE=mtp: Qwen3.8-27B NVFP4 + Q5_K head,
# masked regime MTP draft, 262,144 context, 32 sessions), whose draft-graph captures land at the
# boot probe and at the first requests; three greedy requests, then a clean stop. Read out: every
# `draft-session state high-water`, capture-appetite and capture-WARN line, to compare the fold
# with main. usage: unit-a-cap.sh ARM BOOT
set -u
ARM=$1; N=$2
R=$HOME/.cache/memra-1255/gb10
B=$R/bins/$ARM
OUT=$R/runs/cap-$ARM-b$N
mkdir -p "$OUT"
echo "$$" >> "$R/pids.txt"
S=$OUT/summary.txt
{ date -Is; hostname; sha256sum "$B/memra-server" | sed "s|$HOME|~|"; free -m | head -n 2; } > "$S"
( OUT=$OUT BIN=$B MODE=mtp bash "$B/research/unified-readers-20261114/tools/serve1307.sh" ) > "$OUT/serve.out" 2>&1 &
SP=$!
echo "$SP" >> "$R/pids.txt"
t0=$(date +%s)
if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --timeout 15m --every 5s --quiet; then
  echo "ready after $(( $(date +%s) - t0 )) s" >> "$S"
  for i in 1 2 3; do
    curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"q38","messages":[{"role":"user","content":"Write a short paragraph about the history of the printing press."}],"max_tokens":128,"temperature":0}' \
      > "$OUT/req$i.json" 2>&1
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("req", sys.argv[2], d.get("usage"), d["choices"][0]["finish_reason"])' "$OUT/req$i.json" "$i" >> "$S" 2>&1
  done
else
  echo "NOT READY within 15m" >> "$S"
fi
SPID=$(cat "$OUT/server.pid" 2>/dev/null)
[ -n "$SPID" ] && kill -TERM "$SPID" 2>/dev/null
waitfor --gone "$SP" --timeout 3m --quiet || { [ -n "$SPID" ] && kill -KILL "$SPID" 2>/dev/null; }
grep -E 'draft-session state high-water|capture appetite floor|capture attempt ran out of memory|insufficient VRAM headroom for capture|out of memory' "$OUT/server.log" | cut -c1-400 >> "$S"
echo "CAP-DONE $ARM b$N $(date -Is)" >> "$S"
