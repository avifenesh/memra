#!/bin/bash
# usage: cell-ppn.sh <tag> [extra env assignments...]
# Restarts THIS LANE's memra-server warm with the cell env + this arm's extras.
#
# PID-VERIFIED, never name-matched. This box is shared with the epilogue lane, whose server is
# also called memra-server; `pkill -x memra-server` would be a cross-lane kill. This script only
# signals a PID it wrote, and only after reading /proc/<pid>/cmdline back and confirming the
# binary is mine. If the pidfile names anything else it refuses rather than guessing.
# Own port (18401) so it cannot collide with the epilogue lane's 18400 either.
set -u
TAG=$1; shift
ROOT=$HOME/memra-ppn
BIN=$ROOT/target/release/memra-server
PIDFILE=$HOME/.memra-ppn-server.pid
LOG=$HOME/ppn-cell/cell-$TAG.log

stop_mine() {
    [ -f "$PIDFILE" ] || return 0
    local pid; pid=$(cat "$PIDFILE" 2>/dev/null)
    [ -n "${pid:-}" ] || { rm -f "$PIDFILE"; return 0; }
    if [ ! -r "/proc/$pid/cmdline" ]; then rm -f "$PIDFILE"; return 0; fi
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "memra-ppn/target/release/memra-server"; then
        echo "REFUSING to signal pid $pid: /proc/$pid/cmdline is not my server." >&2
        tr '\0' ' ' < "/proc/$pid/cmdline" >&2; echo >&2
        exit 3
    fi
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 60); do [ -d "/proc/$pid" ] || break; sleep 1; done
    if [ -d "/proc/$pid" ]; then
        echo "pid $pid ignored TERM after 60s; sending KILL (still PID-verified as mine)" >&2
        kill -KILL "$pid" 2>/dev/null; sleep 3
    fi
    rm -f "$PIDFILE"
}

stop_mine
mkdir -p "$HOME/ppn-cell"
: > "$LOG"
# Defaults first, then "$@": `env` lets a later assignment override an earlier one, so the caller
# can override any default while omitting a var still gets it.
env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SPILL_STATS=1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18401 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 "$@" \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$PIDFILE"
echo "started pid $SPID, log $LOG"
