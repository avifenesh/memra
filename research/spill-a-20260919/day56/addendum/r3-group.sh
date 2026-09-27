#!/usr/bin/env bash
# DAY56 section 3: one run of a test group (each --exact) in ONE process with default threads, inside a scope at
# CPUQuota=100% beside sixteen burners. usage: r3-group.sh <binary> <log> <test>...
set -uo pipefail
bin=$1; log=$2; shift 2
here=$(dirname "$0")
pids=()
for _ in $(seq 1 16); do python3 "$here/../../day57/burn.py" & pids+=($!); done
sleep 0.2
"$bin" --exact "$@" > "$log" 2>&1
rc=$?
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null
exit $rc
