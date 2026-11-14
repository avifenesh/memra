#!/usr/bin/env bash
# Runs ON a GB10 host (memra#1316): stop a campaign of this lane early. Ends the campaign script
# CAMPAIGN (so it writes no done file), the running probe_cell.sh, the probe it is waiting on, and
# the readers the cell NAME recorded in reader.pids; then saves the cell's NVRM lines. Stops only
# processes this lane started: they are found by this lane's script paths, never by a loose pattern.
# usage: stop_cell.sh CAMPAIGN_SCRIPT_NAME CELL_NAME     e.g. stop_cell.sh campaign_a.sh a2-live2-exit
set -u
CAMP=$1; NAME=$2
T=research/gb10-boot-oom-20261114/tools
OUT=$HOME/.cache/memra-1316/runs/$NAME
mine() { ps -u "$(id -u)" -o pid=,args= | awk -v pat="$1" 'index($0, pat) && !index($0, "stop_cell.sh") {print $1}'; }
for p in $(mine "$T/$CAMP"); do kill "$p" 2>/dev/null && echo "stopped campaign script $p"; done
for p in $(mine "$T/probe_cell.sh"); do kill "$p" 2>/dev/null && echo "stopped cell script $p"; done
touch "$OUT/STOP"
for p in $(mine "$T/ctx_probe.py"); do kill "$p" 2>/dev/null && echo "stopped probe $p"; done
if [ -s "$OUT/reader.pids" ]; then
  for p in $(cat "$OUT/reader.pids"); do
    for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do
      for g in $(ps -o pid= --ppid "$c" 2>/dev/null); do kill "$g" 2>/dev/null; done
      kill "$c" 2>/dev/null
    done
    kill "$p" 2>/dev/null && echo "stopped reader $p"
  done
fi
t0=$(date -d "$(head -n 1 "$OUT/run.log" | cut -d' ' -f1)" +%s)
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$OUT/nvrm.log"
echo "$(date -Is) stopped early: boots=$(wc -l < "$OUT/boots.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log")" | tee -a "$OUT/run.log"
ps -u "$(id -u)" -o pid=,args= | grep -E "ctx_probe|probe_cell|campaign_|fill_live|timeout 30 cat" | grep -v -E "grep|stop_cell" || echo "nothing of this lane is left running"
