#!/bin/bash
# Yield the 5090: stop this lane's running cell (cell.sh, its arm, the server and the harness) so the gpulock lease
# ends. Touches only processes whose command line names this lane's paths.
set -u
list() { ps -eo pid,args | awk -v pat="$1" 'index($0, pat) && !/awk/ {print $1}'; }
for pat in "rtx5090-above-floor/harness/cell.sh" "rtx5090-above-floor/harness/arm.sh" "memra-reclaim-5090/bins/memra-server-" "prefix-budget-20261107/harness/limit.py 18197" "prefix-budget-20261107/harness/mixed.py 18197"; do
  p=$(list "$pat"); echo "$pat: ${p:-none}"; [ -n "$p" ] && kill -TERM $p
done
waitfor --until "! ps -eo args | grep -q '[m]emra-reclaim-5090/bins/memra-server-'" --timeout 90s --every 2s --quiet || echo "server still alive"
waitfor --until "! ps -eo args | grep -q '[r]tx5090-above-floor/harness/'" --timeout 60s --every 2s --quiet || echo "cell still alive"
gpulock status | head -4
