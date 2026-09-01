#!/bin/bash
# PID-VERIFIED restart of memra-server for the 1M-context demonstration cell
# (lane/glm53-1m-demo). NEVER pkill: a basename kill orphans a VRAM-holding server and can
# hit an unrelated process (GATE:gate-stop-pkill-basename-trap). Adapted from the ring-sizing
# lane's serve.sh; the stop() body is unchanged.
#
# Cell-wide pins (fleet law + this cell's charter):
#   MEMRA_PREFIX_CACHE_MB=0   glm5-specific restore bug, defence in depth (prefix-restore lane)
#   MEMRA_ST_PINNED=1         whole expert set fits pinnable RAM (177.6 GB vs 1007 GB)
#   MEMRA_MOE_FUSED_EPI=1     fused MoE epilogue, measured ~2x prefill (epilogue lane)
#   MEMRA_TIMEOUT_MS_MAX      measurement-cell deadline override (FLAGS.md row, this lane's
#                             commit): the 90 s ceiling is a platform fact of the FRONTED
#                             route; this box is direct-to-server and the cell's ~1M prime
#                             legitimately runs for hours. 64800000 ms = 18 h.
#   MEMRA_MAX_SESSIONS=1      capacity cell: one 1M-plane session at a time; a second
#                             concurrent session must queue, never OOM the demonstration.
#   NVIDIA_TF32_OVERRIDE=0    exactness law.
# Placement (PP4 vs door-off) and MEMRA_CTX come from the caller as extra env assignments.
#
# Usage: serve-1m.sh <tag> <binary-path> [extra env assignments...]
set -u
TAG=$1; BIN=$2; shift 2
SELF=$$
R=${LANE_DIR:-$HOME/lane-1mdemo-vast-20260829}
mkdir -p "$R"

stop() {
  for pid in $(pgrep -x memra-server 2>/dev/null); do
    [ "$pid" = "$SELF" ] && continue
    exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || continue
    case "$exe" in
      */memra-server) ;;
      *) echo "  skip pid $pid: exe=$exe is not a memra-server"; continue ;;
    esac
    echo "  stopping pid $pid exe=$exe"
    kill "$pid"
    for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      exe2=$(readlink -f /proc/$pid/exe 2>/dev/null)
      if [ "$exe2" = "$exe" ]; then
        echo "  escalating SIGKILL on $pid (same pid, same exe)"; kill -9 "$pid"; sleep 3
      else
        echo "  pid $pid was recycled, NOT escalating"
      fi
    fi
  done
}

stop
LOG=$R/serve-$TAG.log
: > "$LOG"
env "$@" MEMRA_PREFIX_CACHE_MB=0 \
  MEMRA_SPILL_STATS=1 MEMRA_ST_PINNED=1 MEMRA_MOE_FUSED_EPI=1 \
  MEMRA_TIMEOUT_MS_MAX=64800000 \
  MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18500 \
  MEMRA_MAX_SESSIONS=1 NVIDIA_TF32_OVERRIDE=0 \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
disown
for i in $(seq 1 900); do
  grep -q "listening on" "$LOG" && break
  grep -qE "panicked" "$LOG" && break
  sleep 2
done
if ! grep -q "listening on" "$LOG"; then
  echo "LOAD FAILED after $((i*2))s"; tail -40 "$LOG"; exit 1
fi
echo "LOAD $TAG ready after ~$((i*2))s  binary=$BIN"
NEW=$(pgrep -x memra-server | head -1)
echo "  pid=$NEW exe=$(readlink -f /proc/$NEW/exe) sha=$(sha256sum $BIN | cut -c1-16)"
echo "--- prefix-cache line (MUST say off / budget 0) ---"
grep -iE "prefix-cache" "$LOG" | head -3
echo "--- caps line ---"
grep -E "template caps" "$LOG" | head -2
echo "--- pp / residency lines ---"
grep -iE "pp |pipeline|stage|resident" "$LOG" | head -12
echo "--- vram after load ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
