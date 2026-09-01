#!/usr/bin/env bash
# PID-VERIFIED stop + start. Never pkill (GATE:gate-stop-pkill-basename-trap): resolve
# /proc/<pid>/exe, refuse anything that is not a memra-server, wait, and escalate only
# after re-verifying the SAME pid AND the SAME exe.
set -uo pipefail
BIN="$1"; LOG="$2"; shift 2
stop() {
  for pid in $(pgrep -f 'memra-server' || true); do
    [ "$pid" = "$$" ] && continue
    exe=$(readlink -f /proc/$pid/exe 2>/dev/null || true)
    case "$exe" in
      */memra-server) ;;
      *) continue ;;
    esac
    echo "[serve] SIGTERM pid=$pid exe=$exe"
    kill -TERM "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      exe2=$(readlink -f /proc/$pid/exe 2>/dev/null || true)
      if [ "$exe2" = "$exe" ]; then echo "[serve] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
    fi
  done
}
stop
: > "$LOG"
env "$@" nohup "$BIN" >> "$LOG" 2>&1 &
echo "[serve] started pid=$! bin=$BIN"
for _ in $(seq 1 400); do
  if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q glm-5.3-flash; then
    echo "[serve] READY after ${SECONDS}s"; exit 0
  fi
  sleep 2
done
echo "[serve] NOT READY"; tail -20 "$LOG"; exit 1
