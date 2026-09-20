#!/usr/bin/env bash
# tools/port-guard.sh — pre-flight port occupancy guard for every gate that boots a server.
#
# WHY THIS FILE EXISTS (GATE-INTEGRITY-20260819 A-16). Seven serving gates bound a fixed port
# with no occupancy check, and two of them bound the SAME one (apikeys-gate.sh and
# serve-st-gate.sh both on 8178), against a receipt that tools/accept-gate.sh already documents
# verbatim at its own guard:
#
#   "the rig's idle llama-server happened to hold 8181, so /health answered INSTANTLY
#    ("up in 0s") from a foreign process, our own server was never waited for, and all six
#    cells failed with HTTP 500 from a server that does not speak this API. Had that foreign
#    process instead answered 200 with a plausible body, the gate would have measured SOMEONE
#    ELSE'S MODEL and pinned it."
#
# That is the shape: not a red gate, a gate that measures the wrong program and reports a
# number. An occupied port is therefore a HARD ABORT, never a wait and never a retry — we
# cannot prove the responder is ours, and "probably ours" is not a gate.
#
# Two functions, and the second one matters as much as the first:
#   memra_port_guard <gate> <port> [<override-env-var>]   pre-flight: refuse if LISTENing
#   memra_port_owned <gate> <port> <pid>                  post-boot: the responder must BE ours
#
# The pre-flight check alone loses to a race — a process that grabs the port in the seconds
# between the check and our bind gets measured. `memra_port_owned` closes that window by
# asserting the listener's pid, which is why accept-gate.sh calls its own version "belt and
# braces". Both are cheap; a gate that boots a server should call both.
#
# FAIL CLOSED ON A MISSING TOOL. If neither `ss` nor `lsof` is available we cannot observe the
# port, and an unobservable port must not read as a free one — that is the fallback-then-claim
# shape this whole audit is about. The guard refuses and names the package.
#
# Usable three ways:
#   source tools/port-guard.sh                       then call the functions
#   tools/port-guard.sh check <gate> <port> [<var>]   standalone (exit 0 free, 1 busy, 2 blind)
#   tools/port-guard.sh listeners <port>              print what holds it
#
# shellcheck shell=bash

# memra_port_listeners <port> -> prints the listener lines (best effort, with pids when the
# kernel lets us see them). Never used for a decision; only for the human reading the refusal.
memra_port_listeners() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | grep "[:.]$port " || ss -tln 2>/dev/null | grep "[:.]$port "
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null
    fi
    return 0
}

# memra_port_observable -> 0 when we have a tool that can see listening sockets.
memra_port_observable() {
    command -v ss >/dev/null 2>&1 || command -v lsof >/dev/null 2>&1
}

# memra_port_busy <port> -> 0 when something is LISTENing on it.
#
# The `[:.]$port ` anchor is the one accept-gate.sh proved out: it matches `0.0.0.0:8178 ` and
# `[::]:8178 ` and `127.0.0.1:8178 ` while refusing to match 18178 or 81780, and the trailing
# space keeps a REMOTE address column from matching a local port.
memra_port_busy() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -tln 2>/dev/null | grep -q "[:.]$port "
    else
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    fi
}

# memra_port_guard <gate> <port> [<override-env-var>] -> 0 free, 1 busy, 2 cannot observe.
memra_port_guard() {
    local gate=$1 port=$2 var=${3:-}
    if ! memra_port_observable; then
        echo "$gate: FAIL — cannot observe listening sockets (no ss, no lsof), so this run"
        echo "  cannot prove port $port is free. An unobservable port is not a free port:"
        echo "  a foreign responder would be measured as the model under test."
        echo "  Install iproute2 (ss) or lsof."
        return 2
    fi
    if memra_port_busy "$port"; then
        echo "$gate: FAIL — port $port is already LISTENing before we start a server."
        memra_port_listeners "$port" | sed 's/^/    /'
        echo "  Refusing to run rather than producing a confused result: a foreign responder on"
        echo "  our port can answer /health, /readyz or /v1/models and be measured as if it were"
        echo "  the model under test (accept-gate.sh:143 records exactly that incident)."
        if [ -n "$var" ]; then
            echo "  Free the port, or set $var=<free port>."
        else
            echo "  Free the port before re-running."
        fi
        return 1
    fi
    return 0
}

# memra_port_owned <gate> <port> <pid> -> 0 when the listener on <port> is <pid>.
#
# Returns 0 (with a NOTE) when pids are not visible to us — `ss -tlnp` needs privilege for
# foreign sockets but always shows our own children, so an invisible pid on our own port is a
# platform quirk, not evidence of a foreign process. It says so out loud rather than claiming
# ownership silently.
memra_port_owned() {
    local gate=$1 port=$2 pid=$3 lines
    command -v ss >/dev/null 2>&1 || return 0
    lines=$(ss -tlnp 2>/dev/null | grep "[:.]$port " || true)
    if [ -z "$lines" ]; then
        echo "$gate: NOTE — no listener visible on $port while checking ownership; skipping the"
        echo "  ownership assertion (the health probe is the authority here)."
        return 0
    fi
    if ! printf '%s\n' "$lines" | grep -q 'pid='; then
        echo "$gate: NOTE — listener pids are not visible on this box; ownership of port $port"
        echo "  not asserted (pre-flight guard still applied)."
        return 0
    fi
    if printf '%s\n' "$lines" | grep -q "pid=$pid,"; then
        return 0
    fi
    echo "$gate: FAIL — port $port answers but is NOT owned by our server (pid $pid)."
    printf '%s\n' "$lines" | sed 's/^/    /'
    echo "  A process that took the port after the pre-flight check would otherwise be measured"
    echo "  as the model under test."
    return 1
}

# Standalone entry point. Sourcing this file executes nothing (BASH_SOURCE != $0).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        check)
            [ $# -ge 3 ] || { echo "usage: port-guard.sh check <gate> <port> [<env-var>]" >&2; exit 2; }
            memra_port_guard "$2" "$3" "${4:-}"
            ;;
        listeners)
            [ $# -ge 2 ] || { echo "usage: port-guard.sh listeners <port>" >&2; exit 2; }
            memra_port_listeners "$2"
            ;;
        *)
            echo "usage: port-guard.sh check <gate> <port> [<env-var>] | listeners <port>" >&2
            exit 2
            ;;
    esac
fi
