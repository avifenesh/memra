#!/usr/bin/env bash
# Probe for waitfor --until (memra#1221 part 2, the queue wrappers): exit 0 when PID has ended, or
# when our ticket (REASON) is still WAITING on the lock and a lease or ticket whose reason names a
# landing or a release shows: the caller then withdraws its waiter so the landing goes first.
# usage: withdraw_probe.sh PID REASON STATUS_CMD...   (e.g. gpulock status / sparklock status --unit u)
pid=$1; reason=$2; shift 2
kill -0 "$pid" 2>/dev/null || exit 0
s=$("$@" 2>/dev/null) || exit 1
echo "$s" | grep -i 'waiting:' | grep -qF "$reason" || exit 1
echo "$s" | grep -qiE 'memra landing|release'
