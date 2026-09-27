#!/usr/bin/env bash
# DAY57 R3: one run of one test inside a scope at CPUQuota=100% that also runs 16 burners, so the test's
# threads wait for CPU as they did beside 900 tests on 48 threads. usage: r2.sh <binary> <test> <log>
set -uo pipefail
bin=$1; t=$2; log=$3
here=$(dirname "$0")
pids=()
for _ in $(seq 1 16); do python3 "$here/burn.py" & pids+=($!); done
sleep 0.2
"$bin" --exact "$t" > "$log" 2>&1
rc=$?
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null
exit $rc
