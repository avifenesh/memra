#!/bin/bash
# Stop this lane's running campaign on GPU 0 (its B binary is being replaced): the campaign shells first so no next
# arm starts, then power-cell.sh (its TERM trap kills the arm's process group and restores the card's limit), then
# the harness (its own process group under timeout). Touches only processes whose command line names this lane.
set -u
R=<lane>
list() { ps -eo pid,args | awk -v pat="$1" 'index($0, pat) && !/awk/ {print $1}'; }
camp=$(list "bash ./campaign.sh 0 $R/out")
echo "campaign: ${camp:-none}"; [ -n "$camp" ] && kill -TERM $camp
pc=$(list "power-cell.sh 0 280 limit-")
echo "power-cell: ${pc:-none}"; [ -n "$pc" ] && kill -TERM $pc
waitfor --until "! ps -eo args | grep -q '[m]emra-server-24f0f95825'" --timeout 90s --every 2s --quiet || echo "server still alive"
h=$(list "$R/harness/")
echo "harness: ${h:-none}"; [ -n "$h" ] && kill -TERM $h
waitfor --until "! ps -eo args | grep -q '[p]ower-cell.sh 0 280'" --timeout 60s --every 2s --quiet || echo "power-cell still alive"
echo "left: $(ps -eo args | grep -c "[m]emra-reclaim/\(harness\|arm.sh\|bins\)") lane processes"
nvidia-smi -i 0 --query-gpu=index,memory.used,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
tail -n 3 "$R/out/power/power-limit-2-B.txt" 2>/dev/null
echo "$(TZ=Asia/Jerusalem date '+%F %T') reclaim: campaign stopped in limit 2 B (review fixes change B)" >> "$R/status.txt"
