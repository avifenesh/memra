#!/bin/bash
# L2 three-arm serve: PID-VERIFIED restart, never name-matched (gate-stop-pkill trap).
# BOX-AB placement (R2 residency shape) + the mandatory pins:
#   MEMRA_PREFIX_CACHE_MB=0, NVIDIA_TF32_OVERRIDE=0, CUDA_VISIBLE_DEVICES=0,1,2
# usage: serve.sh <tag> [extra env...]   -> logs to ~/l3-ab/serve-<tag>.log
set -u
TAG=$1; shift
BIN=$HOME/memra/target/release/memra-server
PIDFILE=$HOME/l3-ab/server.pid
LOG=$HOME/l3-ab/serve-$TAG.log

stop_mine() {
  [ -f "$PIDFILE" ] || return 0
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null)
  [ -n "${pid:-}" ] || { rm -f "$PIDFILE"; return 0; }
  if [ ! -r "/proc/$pid/cmdline" ]; then rm -f "$PIDFILE"; return 0; fi
  if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "memra/target/release/memra-server"; then
    echo "REFUSING to signal pid $pid: not my server" >&2
    tr '\0' ' ' < "/proc/$pid/cmdline" >&2; echo >&2
    exit 3
  fi
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 90); do [ -d "/proc/$pid" ] || break; sleep 1; done
  if [ -d "/proc/$pid" ]; then
    echo "pid $pid ignored TERM; SIGKILL (still PID-verified)" >&2
    kill -KILL "$pid" 2>/dev/null; sleep 3
  fi
  rm -f "$PIDFILE"
}

stop_mine
: > "$LOG"
env MEMRA_PREFIX_CACHE_MB=0 MEMRA_SPILL_STATS=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
  MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 \
  CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18402 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 "$@" \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$PIDFILE"
for i in $(seq 1 1200); do
  grep -q "listening on" "$LOG" && break
  grep -qE "panicked" "$LOG" && break
  sleep 2
done
if ! grep -q "listening on" "$LOG"; then
  echo "LOAD FAILED after $((i*2))s"; tail -30 "$LOG"; exit 1
fi
echo "LOAD $TAG ready after ~$((i*2))s"
echo "  pid=$SPID exe=$(readlink -f /proc/$SPID/exe) sha=$(sha256sum $BIN | cut -c1-16)"
echo "  boot receipts:"
grep -m1 -iE "prefix-cache" "$LOG" | sed 's/^/    /'
echo "    slab-populate gate lines: $(grep -c 'slab-populate.*-gate:' "$LOG")"
grep -m2 "resident-experts decision" "$LOG" | sed 's/^/    /'
echo "    bf16-mmv RESIDENT lines: $(grep -c 'bf16-mmv\] RESIDENT' "$LOG")"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    vram /'
