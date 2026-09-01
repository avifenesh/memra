#!/bin/bash
# PID-verified serve wrapper for the 262k 2-card PINNED-RECIPE cell (lane/glm5-262k-2card-receipt).
# Adapted from the 1m-demo lane's serve-1m.sh with ONE structural change, named here:
#   stop() is PIDFILE-SCOPED, never a basename sweep. This box is shared (window lane port
#   18400, card3 lane memra-server-card3 port 18500); a basename sweep already killed a
#   co-tenant's server once tonight (GATE:gate-stop-pkill-basename-trap + the 02:23Z apology
#   line in BOX-QUEUE.md). stop() kills exactly the PID in $R/server.pid and only after
#   readlink /proc/$PID/exe matches the exe path recorded at launch.
# The binary runs under a unique basename (memra-server-262k) so other lanes' sweeps
# cannot match it either.
#
# Cell env = the owner-accepted 2-card PINNED RECIPE, verbatim from the cell brief:
#   MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
#   MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MOE_RESIDENT_GB=98
#   MEMRA_MOE_SLOTS=16 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0
#   NVIDIA_TF32_OVERRIDE=0 MEMRA_COMPAT=openai
# NAMED DEVIATIONS (receipts law):
#   MEMRA_ADDR=127.0.0.1:18600      coordinator port assignment for this queued slot
#                                   (18400 = window lane, 18500 = card3 lane)
#   MEMRA_TIMEOUT_MS_MAX=64800000   the 1m-demo lane's measurement-cell deadline override
#                                   (commit 7cc36698c, FLAGS.md row), value = that lane's
#                                   serve pin (18 h). Required because the pinned engine's
#                                   90 s first-token ceiling is a platform fact of the
#                                   FRONTED route; this cell is direct-to-server and a
#                                   130k/250k cold prime past 90 s would be cancelled by
#                                   the instrument, not by the wall under test. Binary =
#                                   cc718b988 + cherry-pick 7cc36698c ONLY.
#
# Usage: serve-262k.sh start <tag> <binary-path> | serve-262k.sh stop | serve-262k.sh status
set -u
R=/root/out-262k-2c
PIDFILE=$R/server.pid
EXEFILE=$R/server.exe
mkdir -p "$R"

stop() {
  if [ ! -f "$PIDFILE" ]; then echo "stop: no $PIDFILE, nothing this lane owns is running"; return 0; fi
  pid=$(cat "$PIDFILE")
  exp=$(cat "$EXEFILE" 2>/dev/null || echo MISSING-EXEFILE)
  exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || { echo "stop: pid $pid already gone"; rm -f "$PIDFILE"; return 0; }
  case "$exe" in
    "$exp") ;;
    "$exp (deleted)") ;;
    *) echo "stop: REFUSING kill: pid $pid exe=$exe does not match recorded $exp"; return 1 ;;
  esac
  echo "stop: killing pid $pid exe=$exe (pidfile-scoped)"
  kill "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    exe2=$(readlink -f "/proc/$pid/exe" 2>/dev/null)
    if [ "$exe2" = "$exe" ]; then
      echo "stop: escalating SIGKILL on $pid (same pid, same exe)"; kill -9 "$pid"; sleep 3
    else
      echo "stop: pid $pid was recycled, NOT escalating"
    fi
  fi
  rm -f "$PIDFILE"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    pid=$(cat "$PIDFILE")
    echo "running pid=$pid exe=$(readlink -f /proc/$pid/exe 2>/dev/null)"
  else
    echo "not running"
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

case "${1:-}" in
  stop) stop; exit $? ;;
  status) status; exit 0 ;;
  start) ;;
  *) echo "usage: $0 start <tag> <binary>|stop|status"; exit 1 ;;
esac

TAG=$2; SRCBIN=$3
BIN=$R/bin/memra-server-262k
mkdir -p "$R/bin"
cp -f "$SRCBIN" "$BIN"
chmod +x "$BIN"

stop || exit 1
LOG=$R/serve-$TAG.log
: > "$LOG"
env \
  MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 \
  MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 \
  MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
  MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 \
  NVIDIA_TF32_OVERRIDE=0 MEMRA_COMPAT=openai \
  MEMRA_TIMEOUT_MS_MAX=64800000 \
  MEMRA_MODELS="zai/glm-5.3-flash=/root/models/glm53-nvfp4" \
  MEMRA_ADDR=127.0.0.1:18600 \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
disown
sleep 2
# pgrep -x cannot match a >15-char comm; find the pid by exe readlink instead.
PID=""
for p in /proc/[0-9]*; do
  e=$(readlink -f "$p/exe" 2>/dev/null) || continue
  case "$e" in "$BIN"|"$BIN (deleted)") PID=${p#/proc/}; break ;; esac
done
if [ -z "${PID:-}" ]; then echo "LAUNCH FAILED (no memra-server-262k process)"; tail -20 "$LOG"; exit 1; fi
EXE=$(readlink -f "/proc/$PID/exe")
case "$EXE" in "$BIN"|"$BIN (deleted)") ;; *) echo "LAUNCH VERIFY FAILED: pid $PID exe=$EXE"; exit 1 ;; esac
echo "$PID" > "$PIDFILE"
echo "$BIN" > "$EXEFILE"
echo "launched pid=$PID exe=$EXE sha256=$(sha256sum "$BIN" | cut -c1-16)"

for i in $(seq 1 900); do
  grep -q "listening on" "$LOG" && break
  grep -qE "panicked|FATAL" "$LOG" && break
  kill -0 "$PID" 2>/dev/null || break
  sleep 2
done
if ! grep -q "listening on" "$LOG"; then
  echo "LOAD FAILED after ~$((i*2))s"; tail -40 "$LOG"; exit 1
fi
echo "LOAD $TAG ready after ~$((i*2))s binary=$BIN"
echo "--- prefix-cache line (MUST say off / budget 0) ---"
grep -iE "prefix-cache" "$LOG" | head -3
echo "--- pp / residency / slru decision lines ---"
grep -iE "pp |pipeline|stage|resident|slru" "$LOG" | head -20
echo "--- vram at ready ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$R/vram-at-ready-$TAG.csv"
