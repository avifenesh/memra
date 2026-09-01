#!/bin/bash
# PID-VERIFIED restart of memra-server. NEVER pkill: a basename kill orphans a VRAM-holding
# server and can hit an unrelated process (GATE:gate-stop-pkill-basename-trap).
# Adapted from lane/glm53-box serve.sh for the vast 4-card box. ONE mandatory pin:
#   MEMRA_PREFIX_CACHE_MB=0 on EVERY arm. The prefix-restore lane root-caused a glm5-specific
#   restore bug (PrefixEntry carries kv/recur planes only; glm5_next's full-attention layers
#   live in cache.latent, so a restore set pos=N over latent len=0 and produced confident
#   garbage). The merged head carries the guard; the pin is defence in depth, and a context
#   probe's long arms are exactly the requests that would hit a restore.
# Usage: serve.sh <tag> <binary-path> [extra env assignments...]
set -u
TAG=$1; BIN=$2; shift 2
SELF=$$
R=$HOME/lane-ringsizing-vast-20260829

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
# CUDA_VISIBLE_DEVICES=0,1 kept from the box lane's env for comparability with the 08-28
# receipts; the two spare cards stay untouched for whoever holds them.
env "$@" MEMRA_PREFIX_CACHE_MB=0 \
  MEMRA_SPILL_STATS=1 MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 \
  CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18400 \
  MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
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
echo "--- vram ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
