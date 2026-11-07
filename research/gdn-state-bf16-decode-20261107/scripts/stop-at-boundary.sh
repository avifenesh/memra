#!/usr/bin/env bash
# Stop a running gates script at its next step boundary (the next line in its gates.log), then
# kill its process group so gpulock run releases.  usage: stop-at-boundary.sh LOG PGID
set -u
LOG=${1:?log}; PG=${2:?pgid}
n=$(wc -l < "$LOG")
waitfor --until "test \$(wc -l < $LOG) -gt $n" --timeout 30m --every 5s --quiet
kill -TERM -- -"$PG" 2>/dev/null
echo "stopped pgid $PG after line $((n + 1)): $(sed -n "$((n + 1))p" "$LOG")"
