#!/bin/bash
# Timing arms only. The box is shared with the PP lane; two lanes timing on one box invalidates
# both (LAW:interleaved-ab-protocol — box clock drift and contention are exactly what the x5
# interleave cannot absorb). Correctness arms tolerate a co-tenant and do NOT call this.
#
# Exits 0 only when no compute process holds VRAM on either card EXCEPT my own server, and no
# PP-lane gate runner is mid-arm.
set -u
PIDFILE=$HOME/.memra-epi-server.pid
MINE=$(cat "$PIDFILE" 2>/dev/null || echo "none")
BUSY=0
while IFS=, read -r pid mem name; do
    pid=$(echo "$pid" | tr -d ' '); [ -z "$pid" ] && continue
    if [ "$pid" = "$MINE" ]; then continue; fi
    echo "FOREIGN GPU PROCESS: pid=$pid mem=$mem name=$name"
    BUSY=1
done < <(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader)
# Exclude this script, its parent, and anything whose cmdline merely QUOTES the pattern (an ssh
# `bash -c` carrying this very check would otherwise match itself — observed 2026-08-28).
PP=$(pgrep -f "run-ppn-" 2>/dev/null | grep -vx "$$" | grep -vx "$PPID" || true)
PP_REAL=""
for pid in $PP; do
    cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cl" in *idle-check*|*pgrep*) continue;; esac
    PP_REAL="$PP_REAL $pid"
done
if [ -n "${PP_REAL// /}" ]; then
    echo "PP LANE ARM IN FLIGHT:"
    for pid in $PP_REAL; do echo "  pid=$pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | head -c 160)"; done
    BUSY=1
fi
if [ "$BUSY" = "1" ]; then echo "NOT IDLE — a timing arm must not start."; exit 1; fi
echo "IDLE: no foreign GPU process, no PP arm in flight. $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
