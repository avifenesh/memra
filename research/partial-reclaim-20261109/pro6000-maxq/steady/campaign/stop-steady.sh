#!/bin/bash
# Stop this lane's steady campaign on wrx GPU 0: the launcher and campaign shells first, then the tracked wrapper
# (SIGTERM: it stops its whole process group and finishes the lease). Touches only processes naming this lane.
set -u
R=<lane>
W=<home>/.local/bin/waitfor
list() { ps -eo pid,args | awk -v pat="$1" 'index($0, pat) && !/awk/ {print $1}'; }
for pat in "bash ./run-steady.sh" "bash ./campaign-steady.sh $R/out"; do
  p=$(list "$pat"); echo "$pat: ${p:-none}"; [ -n "$p" ] && kill -TERM $p
done
p=$(list "memra-gpu-run --gpus <gpu> --receipt $R/out/lease")
echo "wrapper: ${p:-none}"; [ -n "$p" ] && kill -TERM $p
$W --until "! ps -eo args | grep -q '[m]emra-1124/reclaim/bins/memra-server'" --timeout 120s --every 3s --quiet || echo "server still alive"
$W --until "! ps -eo args | grep -q '[m]emra-1124/reclaim/\(harness\|arm-steady\)'" --timeout 60s --every 3s --quiet || echo "arm still alive"
echo "left: $(ps -eo args | grep -c '[m]emra-1124/reclaim/')"
nvidia-smi -i 0 --query-gpu=index,memory.used,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c f8021f32
echo "$(TZ=Asia/Jerusalem date '+%F %T') steady: stopped in arm 1 (6 workers never reach the reclaim); rerun at 8" >> "$R/status.txt"
