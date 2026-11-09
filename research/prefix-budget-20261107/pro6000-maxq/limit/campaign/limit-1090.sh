#!/bin/bash
# #1090 at-the-limit cell on one RTX PRO 6000 Max-Q (nohup): A = main 9877f6a696, B = #1090 aabb3ae1b6
# (cell_evidence's own release builds of both), four fresh boots interleaved A B B A, each running
# harness/limit.py (fill the cache and the parked pool, pressure at the interactive cap, repeat) at
# MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=12. The card: wrx GPU G under gpuG.lock, no compute on it at start,
# power-cell.sh at the card's 280 W limit (Max-Q thermal stop and throttle log).
#   limit-1090.sh G OUT
set -u
. <lane>/env.sh
G=$1; O=$2; S=$W/status.txt
case $G in 2) UUID=$GPU2_UUID ;; 3) UUID=$GPU3_UUID ;; *) echo "GPU must be 2 or 3"; exit 2 ;; esac
A=$W/bins1090/memra-server-9877f6a696
B=$W/bins1090/memra-server-aabb3ae1b6
mkdir -p "$O" "$O/power"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$O/campaign.log" >> $S; }
waitfor --until "! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q '^$UUID'" --timeout 30m --every 15s --quiet \
  || { log "limit1090: GPU $G still runs compute"; exit 4; }
(
  flock -w 3600 9 || { log "limit1090: gpu$G.lock not granted"; exit 9; }
  log "limit1090: start on GPU $G; $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
  i=0
  for a in A B B A; do
    i=$((i + 1))
    bin=$A; [ $a = B ] && bin=$B
    log "limit1090: $i $a start"
    ARM_GPU_UUID=$UUID ARM_PORT=1829$G bash $M/power/power-cell.sh $G 280 limit-$i-$a $O/power -- \
      bash $W/limit/arm-limit.sh "limit-$i-$a" "$bin" "$O" limit MEMRA_MAX_SESSIONS=12
    log "limit1090: $i $a rc=$? $(grep -a 'reclaim_on_defer_lines\|oom_lines\|admit_oom' "$O/limit-$i-$a/arm.txt" | tr '\n' ' ') $(cut -c1-400 "$O/limit-$i-$a/result.json" 2>/dev/null)"
  done
) 9> <box-locks>/gpu$G.lock
log "limit1090: done OUT=$O"
