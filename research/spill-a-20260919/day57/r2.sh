#!/usr/bin/env bash
# DAY55 R2 (section 3): one run of one test inside a scope at CPUQuota=100% that also runs 8 burners, so the test's
# threads wait for CPU as they did beside 900 tests on 48 threads. usage: r2.sh <binary> <test> <log>
set -uo pipefail
bin=$1; t=$2; log=$3
here=$(dirname "$0")
pids=()
for _ in 1 2 3 4 5 6 7 8; do python3 "$here/burn.py" & pids+=($!); done
sleep 0.2
"$bin" --exact "$t" > "$log" 2>&1
rc=$?
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null
exit $rc
