#!/usr/bin/env bash
# One #1090 campaign on GPU G (nohup), no record wait: gpuG.lock, power-cell.sh at 280 W (the card's set limit,
# one power regime per A/B; Max-Q thermal stop and throttle log; REASSERT=1). Both arms of every phase run on this
# card, interleaved.  usage: run-1090b.sh G TAG PHASES
set -u
. <lane>/env.sh
G=$1; TAG=$2; PHASES=$3
O=$W/p1090/$TAG
mkdir -p "$O/power"
log() { echo "$(TZ=Asia/Jerusalem date -Is) $*" >> "$O/campaign.log"; }
case $G in 0) U=$GPU0_UUID ;; 1) U=$GPU1_UUID ;; 2) U=$GPU2_UUID ;; 3) U=$GPU3_UUID ;; *) exit 2 ;; esac
export ARM_GPU_UUID=$U ARM_PORT=$((18280 + G))
waitfor --until "test -f $W/BUILT1090" --timeout 3h --every 30s --quiet || { log "no #1090 binaries"; exit 7; }
(
  flock -w 14400 9 || { log "gpu$G.lock not granted"; exit 9; }
  log "gpu$G.lock held; $(nvidia-smi -i $G --query-gpu=memory.used,power.limit,temperature.gpu,utilization.gpu --format=csv,noheader)"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$U" && { log "GPU $G busy"; exit 8; }
  REASSERT=1 bash $M/power/power-cell.sh $G 280 p1090-$TAG "$O/power" -- bash $W/phases-1090b.sh "$O" "$PHASES"
  echo $? > "$O/rc"
) 9> <box-locks>/gpu$G.lock
