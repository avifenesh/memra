#!/bin/bash
# Partial-reclaim acceptance on one RTX PRO 6000 Max-Q (nohup, run under taskset -c 24-31,56-63):
# A = main d35337f51a (#1090 landed: the reclaim flush), B = fix/partial-reclaim-discrete ab21e254a6, both release builds
# from one clone and one target dir.
#   1. limit: A B B A, harness/limit.py (fill the cache and the parked pool, pressure at the interactive cap, repeat)
#   2. mixed (regression, counts only): A B B A A B, harness/mixed.py
# Both at MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=12, a fresh boot per arm. The card: GPU G (0 or 1) under gpuG.lock, no
# compute on it at start, power-cell.sh at 280 W with REASSERT=1 (the box re-applies its own cap on a timer); a
# thermal stop (rc 143) ends the campaign.
#   campaign.sh G OUT
set -u
. <kit>/env.sh
R=<lane>
export TMPDIR=$R/tmp
G=$1; O=$2; S=$R/status.txt
case $G in 0) UUID=$GPU0_UUID ;; 1) UUID=$GPU1_UUID ;; *) echo "GPU must be 0 or 1"; exit 2 ;; esac
A=$R/bins/memra-server-d35337f51a
B=$R/bins/memra-server-ab21e254a6
for b in "$A" "$B"; do [ -x "$b" ] || { echo "missing $b"; exit 3; }; done
mkdir -p "$O/limit" "$O/mixed" "$O/power"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$O/campaign.log" >> "$S"; }
waitfor --until "! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q '^$UUID'" --timeout 30m --every 15s --quiet \
  || { log "reclaim: GPU $G still runs compute"; exit 4; }
(
  flock -w 3600 9 || { log "reclaim: gpu$G.lock not granted"; exit 9; }
  log "reclaim: start on GPU $G; affinity $(taskset -pc $$ | cut -d: -f2); $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | tr '\n' ' ')"
  run() { # phase index arm workload
    local ph=$1 i=$2 a=$3 wl=$4 bin=$A rc
    [ "$a" = B ] && bin=$B
    log "reclaim: $ph $i $a start"
    REASSERT=1 ARM_GPU_UUID=$UUID ARM_PORT=1830$G bash "$M/power/power-cell.sh" "$G" 280 "$ph-$i-$a" "$O/power" -- \
      bash "$R/arm.sh" "$ph-$i-$a" "$bin" "$O/$ph" "$wl" MEMRA_MAX_SESSIONS=12
    rc=$?
    log "reclaim: $ph $i $a rc=$rc $(grep -a 'reclaim_on_defer_lines\|oom_lines' "$O/$ph/$ph-$i-$a/arm.txt" | tr '\n' ' ') $(cut -c1-300 "$O/$ph/$ph-$i-$a/result.json" 2>/dev/null)"
    [ "$rc" = 143 ] && { log "reclaim: thermal stop, campaign ends"; exit 143; }
    return 0
  }
  i=0; for a in A B B A; do i=$((i + 1)); run limit "$i" "$a" limit; done
  i=0; for a in A B B A A B; do i=$((i + 1)); run mixed "$i" "$a" mixed; done
) 9> <box-locks>/gpu$G.lock
log "reclaim: done OUT=$O"
