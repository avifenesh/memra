#!/bin/bash
# Stop this lane's waves campaigns on GPUs 0 and 1 (owner cap: 250 W while layby's blocks run). Campaign shells
# first so no next arm starts, then power-cell.sh (its TERM trap kills the arm's group, then it restores the limit it
# found, 250 W), then the harness. Touches only processes whose command line names this lane.
set -u
R=<lane>
W=<home>/.local/bin/waitfor
list() { ps -eo pid,args | awk -v pat="$1" 'index($0, pat) && !/awk/ {print $1}'; }
for pat in "bash ./after-gpu0.sh" "bash ./after-mixed-gpu1.sh" "bash ./campaign-ab.sh 0 " "bash ./campaign-ab.sh 1 "; do
  p=$(list "$pat"); echo "$pat: ${p:-none}"; [ -n "$p" ] && kill -TERM $p
done
pc=$(list "power-cell.sh 0 280 waves-"; list "power-cell.sh 1 280 waves-")
echo "power-cell: ${pc:-none}"; [ -n "$pc" ] && kill -TERM $pc
$W --until "! ps -eo args | grep -q '[m]emra-server-\(d35337f51a\|ab21e254a6\)'" --timeout 90s --every 2s --quiet || echo "server still alive"
h=$(list "$R/harness/")
echo "harness: ${h:-none}"; [ -n "$h" ] && kill -TERM $h
$W --until "! ps -eo args | grep -q '[p]ower-cell.sh [01] 280'" --timeout 60s --every 2s --quiet || echo "power-cell still alive"
echo "left: $(ps -eo args | grep -c "[m]emra-reclaim/\(harness\|arm\|bins\|campaign\)") lane processes"
nvidia-smi -i 0,1 --query-gpu=index,memory.used,power.limit --format=csv,noheader
ls <power>/
echo "$(TZ=Asia/Jerusalem date '+%F %T') reclaim: waves campaigns stopped for the 250 W cap" >> "$R/status.txt"
