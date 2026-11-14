#!/usr/bin/env bash
# Runs ON a GB10 host (memra#1316): stop a serve_cycle.sh cell NAME early. Ends the campaign and
# cell scripts, the server of the running cycle (SIGTERM, its own shutdown), the cycle's readers
# and the memory watcher; then saves the cell's NVRM lines. Stops only this lane's processes,
# found by this lane's script and binary paths.
# usage: stop_cycle.sh CELL_NAME
set -u
NAME=$1
T=research/gb10-boot-oom-20261114/tools
CD=$HOME/.cache/memra-1316/runs/$NAME
mine() { ps -u "$(id -u)" -o pid=,args= | awk -v pat="$1" 'index($0, pat) && !index($0, "stop_cycle.sh") {print $1}'; }
for pat in "$T/campaign_c.sh" "$T/serve_cycle.sh"; do
  for p in $(mine "$pat"); do kill "$p" 2>/dev/null && echo "stopped script $p"; done
done
touch "$CD/memwatch.stop"
for d in "$CD"/*/; do
  touch "$d/stop"
  if [ -s "$d/reader.pids" ]; then
    for p in $(cat "$d/reader.pids"); do
      for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do
        for g in $(ps -o pid= --ppid "$c" 2>/dev/null); do kill "$g" 2>/dev/null; done
        kill "$c" 2>/dev/null
      done
      kill "$p" 2>/dev/null
    done
  fi
done
for p in $(mine ".cache/memra-1316/bins/"); do kill -TERM "$p" 2>/dev/null && echo "SIGTERM server $p"; done
waitfor --until "! ps -u $(id -u) -o args= | grep -q '[.]cache/memra-1316/bins/'" --every 2s --timeout 2m --quiet
for p in $(mine "$T/serve1316.sh") $(mine "$T/client1316.py") $(mine "$T/fill_live.sh"); do kill "$p" 2>/dev/null; done
t0=$(date -d "$(head -n 1 "$CD/run.log" | cut -d' ' -f1)" +%s)
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$CD/nvrm.log"
echo "$(date -Is) stopped early: cycles=$(wc -l < "$CD/cycles.jsonl") nvrm_lines=$(wc -l < "$CD/nvrm.log")" | tee -a "$CD/run.log"
ps -u "$(id -u)" -o pid=,args= | grep -E "memra-1316|timeout 30 cat" | grep -v -E "grep|stop_cycle" || echo "nothing of this lane is left running"
