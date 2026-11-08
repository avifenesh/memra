#!/usr/bin/env bash
# The #1090 campaign on GPU 3 (nohup). GPU 2 carries the landing agent's rtxpro6000 record (moved there
# 19:55) and GPUs 0,1 the DSv4 pair cells (pass/fail on identity). One scored campaign per box: this timed A/B
# waits until the record's cell_evidence run has started and ended (its perf phase is the record's last part).
# gpu3.lock (box convention), power-cell.sh at 280 W (the card's set limit; no 300 W raise, so the A/B keeps one
# power regime), with its Max-Q thermal stop (>= 95 C, or >= 92 C with fan >= 95 % on 3 consecutive samples) and
# throttle log; REASSERT=1 holds the limit against the box's cap timer.
# usage: run-1090.sh TAG [PHASES]
set -u
. <lane>/env.sh
TAG=$1; PHASES=${2:-m12 mdef c1 c4}
O=$W/p1090/$TAG
mkdir -p "$O/power"
log() { echo "$(TZ=Asia/Jerusalem date -Is) $*" >> "$O/campaign.log"; }
REC='cell_evidence[.]py run .*--device rtxpro6000'
if waitfor --until "pgrep -f '$REC' > /dev/null" --timeout 3h --every 30s --quiet; then
  log "record running: $(pgrep -a -f "$REC" | head -n 1 | cut -c1-200)"
  waitfor --no-proc "$REC" --timeout 4h --every 30s --quiet || { log "record still running after 4 h"; exit 7; }
  log "record over"
else
  log "no rtxpro6000 record process seen in 3 h; going ahead"
fi
waitfor --until "test -f $W/BUILT1090" --timeout 3h --every 30s --quiet || { log "no #1090 binaries"; exit 7; }
(
  flock -w 14400 9 || { log "gpu3.lock not granted"; exit 9; }
  log "gpu3.lock held; $(nvidia-smi -i 3 --query-gpu=memory.used,power.limit,temperature.gpu,utilization.gpu --format=csv,noheader)"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$GPU3_UUID" && { log "GPU 3 busy"; exit 8; }
  waitfor --until "test \"\$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')\" -le 5" \
    --timeout 10m --every 10s --quiet || { log "GPU 3 does not read idle"; exit 8; }
  REASSERT=1 bash $M/power/power-cell.sh 3 280 p1090-$TAG "$O/power" -- bash $W/phases-1090.sh "$O" "$PHASES"
  echo $? > "$O/rc"
) 9> <box-locks>/gpu3.lock
