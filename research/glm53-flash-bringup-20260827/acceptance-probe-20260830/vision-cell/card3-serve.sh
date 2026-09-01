#!/bin/bash
# card3-lane serve: 1-card SLRU boot of memra-server on CUDA_VISIBLE_DEVICES=3, port 18500.
# NEVER a serving claim; probe posture only.
# PID-VERIFIED stop, pidfile-scoped: this script only ever signals the pid it launched
# (recorded in /root/card3-lane/serve.pid) after verifying /proc/<pid>/exe is OUR binary
# under /root/memra-card3/target. Other agents' memra-server processes are never touched.
# Usage: serve-card3.sh start <tag> [extra env...] | serve-card3.sh stop | serve-card3.sh status
set -u
LANE=/root/card3-lane
# DISTINCT BASENAME (co-tenancy incident 2026-08-30 ~02:3x): the timed window's runbook
# serve.sh stop() kills every pid named memra-server whose exe ends */memra-server —
# including this lane's. A distinct basename keeps this server out of their sweep;
# stopping it remains THIS script's pidfile-verified job alone.
BIN=/root/card3-lane/bin/memra-server-card3
PIDFILE=$LANE/serve.pid

stop_mine() {
  [ -f "$PIDFILE" ] || { echo "no pidfile, nothing to stop"; return 0; }
  local pid exe
  pid=$(cat "$PIDFILE")
  exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || { echo "pid $pid gone"; rm -f "$PIDFILE"; return 0; }
  case "$exe" in
    /root/card3-lane/bin/memra-server-card3) ;;
    *) echo "REFUSING: pid $pid exe=$exe is not our binary"; return 1 ;;
  esac
  echo "stopping pid $pid exe=$exe"
  kill "$pid"
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    local exe2
    exe2=$(readlink -f /proc/$pid/exe 2>/dev/null)
    if [ "$exe2" = "$exe" ]; then
      echo "escalating SIGKILL on $pid (same pid, same exe)"; kill -9 "$pid"; sleep 3
    else
      echo "pid recycled, NOT escalating"
    fi
  fi
  rm -f "$PIDFILE"
}

case "${1:-}" in
  stop) stop_mine; exit $? ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      pid=$(cat $PIDFILE)
      echo "running pid=$pid exe=$(readlink -f /proc/$pid/exe)"
    else
      echo "not running"
    fi
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 start <tag> [env...] | stop | status"; exit 1 ;;
esac

shift
TAG=$1; shift
stop_mine || exit 1
LOG=$LANE/logs/serve-$TAG.log
: > "$LOG"
env "$@" \
  MEMRA_SPILL_STATS=1 MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0 \
  CUDA_VISIBLE_DEVICES=3 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=/root/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18500 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=2 NVIDIA_TF32_OVERRIDE=0 \
  nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown
sleep 2
EXE=$(readlink -f /proc/$PID/exe 2>/dev/null || true)
if [ "$EXE" != "$(readlink -f $BIN)" ]; then
  # nohup may have forked on some shells; find OUR binary's pid by exact exe path.
  PID=""
  for p in $(pgrep -x memra-server-card3); do
    [ "$(readlink -f /proc/$p/exe 2>/dev/null)" = "$(readlink -f $BIN)" ] && PID=$p
  done
  [ -n "$PID" ] || { echo "LAUNCH FAILED: our binary not found in process table"; tail -20 "$LOG"; exit 1; }
fi
echo "$PID" > "$PIDFILE"
for i in $(seq 1 900); do
  grep -q 'listening on' "$LOG" && break
  grep -qE 'panicked|FATAL' "$LOG" && break
  kill -0 "$PID" 2>/dev/null || break
  sleep 2
done
if ! grep -q 'listening on' "$LOG"; then
  echo "LOAD FAILED after ~$((i*2))s"; tail -30 "$LOG"; rm -f "$PIDFILE"; exit 1
fi
echo "LOAD $TAG ready after ~$((i*2))s pid=$PID exe=$(readlink -f /proc/$PID/exe)"
echo "--- vram (card 3 = index 3) ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i 3
FREE_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 3)
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
echo "free on card3: $((FREE_MIB-USED_MIB)) MiB"
