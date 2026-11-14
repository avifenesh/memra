#!/usr/bin/env bash
# Runs ON a GB10 host (memra#1316): stop a boot_cell.sh cell NAME early, e.g. before the lease
# ends. Ends the process group recorded in jobs/JOB.pid (the campaign script, boot_cell.sh,
# server_boot.py, the readers, the memory watcher), then any server of this lane's binaries still
# booting (each runs in its own process group), and saves the cell's NVRM lines and summary line.
# usage: stop_boot.sh JOB CELL_NAME
set -u
JOB=$1; NAME=$2
R=$HOME/.cache/memra-1316
OUT=$R/runs/$NAME
g=$(cat "$R/jobs/$JOB.pid")
kill -TERM -- "-$g" 2>/dev/null && echo "stopped group $g"
touch "$OUT/STOP"
for p in $(ps -u "$(id -u)" -o pid=,args= | awk 'index($0, ".cache/memra-1316/bins/") && index($0, "memra-server") {print $1}'); do
  kill -KILL "$p" 2>/dev/null && echo "killed server $p"
done
t0=$(date -d "$(head -n 1 "$OUT/run.log" | cut -d' ' -f1)" +%s)
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$OUT/nvrm.log"
echo "$(date -Is) stopped early: boots=$(wc -l < "$OUT/boots.jsonl") died=$(grep -c '"outcome": "fatal"' "$OUT/boots.jsonl") first_try_failed=$(grep -c '"first_try_failed": true' "$OUT/boots.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log") kgrctx=$(grep -c kgrctxAllocCtxBuffers "$OUT/nvrm.log")" | tee -a "$OUT/run.log"
python3 -c "import time; time.sleep(1)"
ps -u "$(id -u)" -o pid=,args= | grep -E "memra-1316|timeout 30 cat" | grep -v -E "grep|stop_boot" || echo "nothing of this lane is left running"
