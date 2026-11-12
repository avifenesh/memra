#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1221 part 2): warm == cold pair cells
# (box/pairs_1221b.py) on one binary, in their own process group, stopped at CAP_MIN.
# usage: pairs_run.sh OUT_NAME CAP_MIN CASE...   (CASE as pairs_1221b.py; BIN may be bins/<dir>)
set -u
D=$HOME/.cache/memra-1221b; cd "$D" || exit 1
NAME=$1; CAP=$2; shift 2
OUT=$D/runs/pairs/$NAME; mkdir -p "$OUT"
{ date -Is; nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader; uptime; } > "$OUT/guard.log"
setsid python3 box/pairs_1221b.py "$OUT" "$@" > "$OUT/run.log" 2>&1 < /dev/null &
p=$!
if ! waitfor --gone "$p" --timeout "${CAP}m" --quiet; then
  echo "$(date -Is) pairs_run: cap ${CAP} min reached; stopping group $p" >> "$OUT/run.log"
  kill -TERM -"$p"; waitfor --gone "$p" --timeout 1m --quiet || kill -KILL -"$p"
fi
wait "$p"
for f in "$OUT"/*/*.log; do [ -f "$f" ] && gzip -f "$f"; done
