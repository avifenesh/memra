#!/bin/bash
# usage: cell-epi.sh <tag> [extra env assignments...]
# Restarts MY memra-server WARM (never drops caches: a cold reload costs ~22 min, METHOD.txt)
# with the A4BEST serving env + this arm's extras.
#
# DIFFERENCE FROM cell2.sh, and it is not cosmetic: cell2.sh stops the server with
# `pkill -x memra-server`. This box is SHARED with the PP lane (~/memra-ppn), whose gate runner
# can legitimately have a memra binary up, so a name-matched pkill is a cross-lane kill. This
# script only ever signals a PID IT WROTE, and only after reading /proc/<pid>/cmdline back and
# confirming the binary is mine. If the pidfile names something else it refuses and exits rather
# than guessing.
set -u
TAG=$1; shift
ROOT=$HOME/memra-epi
BIN=$ROOT/target/release/memra-server
PIDFILE=$HOME/.memra-epi-server.pid
LOG=$HOME/cell-$TAG.log

stop_mine() {
    [ -f "$PIDFILE" ] || return 0
    local pid; pid=$(cat "$PIDFILE" 2>/dev/null)
    [ -n "${pid:-}" ] || { rm -f "$PIDFILE"; return 0; }
    if [ ! -r "/proc/$pid/cmdline" ]; then rm -f "$PIDFILE"; return 0; fi
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "memra-epi/target/release/memra-server"; then
        echo "REFUSING to signal pid $pid: /proc/$pid/cmdline is not my server." >&2
        tr '\0' ' ' < "/proc/$pid/cmdline" >&2; echo >&2
        exit 3
    fi
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done
    if [ -d "/proc/$pid" ]; then
        echo "pid $pid ignored TERM after 30s; sending KILL (still PID-verified as mine)" >&2
        kill -KILL "$pid" 2>/dev/null; sleep 3
    fi
    rm -f "$PIDFILE"
}

stop_mine
: > "$LOG"
# DEFAULTS FIRST, then "$@": `env` lets a later assignment override an earlier one, so with the
# caller's vars first every hardcoded value below silently WON and a caller could not, for
# example, pin the server to one card. Defaults first means the caller overrides; omitting a var
# still gets the default. (Found while pinning an identity arm to card 1 to stay clear of the
# co-tenant lane's 82 GB peak on card 0.)
env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SPILL_STATS=1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18400 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 "$@" \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$PIDFILE"
disown
for i in $(seq 1 900); do
    grep -q 'listening on' "$LOG" && break
    grep -qE '^\[server\] .*(error|failed)|panicked' "$LOG" && break
    [ -d "/proc/$SPID" ] || break
    sleep 2
done
if ! grep -q 'listening on' "$LOG"; then echo "LOAD FAILED after $((i*2))s"; tail -20 "$LOG"; exit 1; fi
echo "LOAD $TAG: ready after ~$((i*2))s (pid $SPID)"
echo "ARM ENV: $*"
grep -E '\[moe\] resident|moe-cache\] size-aware|decode wave cap|EAGER-ONLY|spill-pread|PP |pp ' "$LOG" | head -10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
