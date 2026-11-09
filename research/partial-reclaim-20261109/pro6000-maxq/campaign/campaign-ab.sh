#!/bin/bash
# Partial-reclaim lane: one workload's A/B on one card in the given order, a fresh boot per arm.
# A = main d35337f51a (#1090 landed), B = fix/partial-reclaim-discrete ab21e254a6. MEMRA_CTX=32768,
# MEMRA_MAX_SESSIONS=12. GPU G under gpuG.lock, no compute on it at start, power-cell.sh at 280 W with REASSERT=1.
# Writes "HOLDS GPU<G> <time>" to $R/gpu<G>.txt once the lock is held. Run under taskset -c 24-31,56-63.
#   campaign-ab.sh G OUT WORKLOAD ORDER   (e.g. 1 OUT waves "A B B A")
set -u
. <kit>/env.sh
R=<lane>
export TMPDIR=$R/tmp
G=$1; O=$2; WL=$3; ORDER=$4; S=$R/status.txt
case $G in 0) UUID=$GPU0_UUID ;; 1) UUID=$GPU1_UUID ;; *) echo "GPU must be 0 or 1"; exit 2 ;; esac
A=$R/bins/memra-server-d35337f51a
B=$R/bins/memra-server-ab21e254a6
for b in "$A" "$B"; do [ -x "$b" ] || { echo "missing $b"; exit 3; }; done
mkdir -p "$O/$WL" "$O/power" "$TMPDIR"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$O/campaign.log" >> "$S"; }
waitfor --until "! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q '^$UUID'" --timeout 30m --every 15s --quiet \
  || { log "reclaim-gpu$G: GPU $G still runs compute"; exit 4; }
(
  flock -w 600 9 || { log "reclaim-gpu$G: gpu$G.lock not granted"; exit 9; }
  echo "HOLDS GPU$G $(TZ=Asia/Jerusalem date '+%F %T') ($WL)" >> "$R/gpu$G.txt"
  log "reclaim-gpu$G: start; affinity $(taskset -pc $$ | cut -d: -f2); $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | tr '\n' ' ')"
  i=0
  for a in $ORDER; do
    i=$((i + 1)); bin=$A; [ "$a" = B ] && bin=$B
    log "reclaim-gpu$G: $WL $i $a start"
    REASSERT=1 ARM_GPU_UUID=$UUID ARM_PORT=1830$G bash "$M/power/power-cell.sh" "$G" 280 "$WL-$i-$a" "$O/power" -- \
      bash "$R/arm-waves.sh" "$WL-$i-$a" "$bin" "$O/$WL" "$WL" MEMRA_MAX_SESSIONS=12
    rc=$?
    log "reclaim-gpu$G: $WL $i $a rc=$rc $(grep -a 'reclaim_on_defer_lines\|oom_lines' "$O/$WL/$WL-$i-$a/arm.txt" | tr '\n' ' ')"
    [ "$rc" = 143 ] && { log "reclaim-gpu$G: thermal stop, campaign ends"; exit 143; }
  done
  echo "RELEASED GPU$G $(TZ=Asia/Jerusalem date '+%F %T') ($WL)" >> "$R/gpu$G.txt"
) 9> <box-locks>/gpu$G.lock
log "reclaim-gpu$G: $WL done OUT=$O"
