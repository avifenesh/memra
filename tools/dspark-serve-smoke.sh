#!/usr/bin/env bash
# dspark-serve-smoke.sh — served-stream identity for the dspark route (lane/dspark-q38-recover).
#
# Boots memra-server twice on the same trunk artifact: once with the dspark route armed
# (MEMRA_DSPARK_SPEC=1 + MEMRA_DSPARK_DRAFT=<export_dir>), once spec-off (MEMRA_SERVE_SPEC=0).
# Asserts, over the real HTTP surface:
#   1. greedy completions byte-identical between the two boots (3 prompt classes);
#   2. every dspark-served row reports positive rounds + drafted tokens, and the server log
#      carries [dspark-acc] lines (the route ENGAGED — acceptance itself may legitimately be 0);
#   3. a sampled request on the dspark boot engages rejection-verify speculation;
#   4. two concurrent greedy requests both draft under the default LOW=2 wave policy;
#   5. a greedy request behind a live sampled DFlash row STACKS into the LOW band. STALE-GATE
#      FIX (mtp-skip lane, 2026-08-30): this tooth used to assert the opposite (greedy stays
#      plain behind one non-demotable sampled row), the posture 62f48ac1c4 hardened on
#      2026-08-24; c4432f4a4 refuted and removed that block ON MEASUREMENT the next day (the
#      LOW band now stacks dspark rows bounded by the wave projection) but did not update this
#      tooth, so the gate has been red-on-main for every binary since. The old solo posture
#      still exists behind the MEMRA_DSPARK_SAMPLED_WAVE=0 seam; this smoke asserts the
#      default.
#
# usage: dspark-serve-smoke.sh <trunk.gguf> <draft_export_dir> <server_bin> <evidence_dir>
set -uo pipefail
MODEL=$1; DRAFT=$2; BIN=$3; EV=$4
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-serve.lock}
PORT=${MEMRA_GATE_PORT:-18097}
mkdir -p "$EV"
SERVER_PID=""
REQUEST_PID=""
FAIL=0

# GATE-INTEGRITY-20260819 A-16. The `curl /v1/models` check below is a RESPONDER check, not an
# occupancy check: it only sees a squatter that speaks this API. A foreign process that answers
# something else, or holds the port without answering at all, walks straight past it — and then
# our own server fails to bind while the boot loop keeps polling whatever is there. The guard
# refuses on ANY listener, and `memra_port_owned` asserts the ready responder is our child.
# Both checks are kept: they fail on different things.
HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE_DIR/port-guard.sh" ] || {
    echo "dspark-serve-smoke: FAIL — $HERE_DIR/port-guard.sh is missing; refusing to bind a"
    echo "  port this run cannot prove is free."
    exit 1
}
# shellcheck source=/dev/null
. "$HERE_DIR/port-guard.sh"

boot() { # $1 extra-env  $2 log
    memra_port_guard dspark-serve-smoke "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving — refusing to boot over it"; return 1
    fi
    command -v setsid >/dev/null 2>&1 || { echo "setsid is required for PID-scoped cleanup"; return 1; }
    # shellcheck disable=SC2086
    setsid flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        $1 "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    # NO memra_port_owned here, and that is a decision rather than an omission: $SERVER_PID is
    # the `flock` WRAPPER's pid (this file's own stop() comment records the same fact), so the
    # listener pid is its child and asserting equality against $SERVER_PID would manufacture a
    # false red. Passing a pid we know is wrong is worse than not asserting.
    #
    # Residual after the pre-flight guard: only a process that grabs the port during the boot
    # window, and it would also have to answer /v1/models to get past the loop below. The private
    # process group makes cleanup exact even though the existing ownership helper expects one pid.
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died"; tail -5 "$2"; return 1; }
        sleep 2
    done
    return 1
}
stop() {
    if [ -n "$REQUEST_PID" ]; then
        kill -TERM "$REQUEST_PID" 2>/dev/null || true
        wait "$REQUEST_PID" 2>/dev/null || true
        REQUEST_PID=""
    fi
    [ -n "$SERVER_PID" ] || return 0
    # setsid makes the flock wrapper and its server child one private process group. Kill only
    # that group: a name-wide pkill can terminate a co-tenant server on the other GPU.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 2
}
trap stop EXIT
trap 'stop; exit 130' INT
trap 'stop; exit 143' TERM
req_n() { # $1 prompt  $2 temp  $3 max_tokens  $4 out.json
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d "{\"model\":\"gate\",\"temperature\":$2,\"max_tokens\":$3,\"messages\":[{\"role\":\"user\",\"content\":$1}]}" \
        >"$4"
}
req() { # $1 prompt  $2 temp  $3 out.json
    req_n "$1" "$2" 96 "$3"
}
spec_engaged() { # $1 response.json
    jq -e '(.usage.spec.rounds // 0) > 0 and (.usage.spec.drafted // 0) > 0' "$1" \
        >/dev/null 2>&1
}
spec_plain() { # $1 response.json
    jq -e '.usage.spec == null' "$1" >/dev/null 2>&1
}
valid_completion() { # $1 response.json
    jq -e '(.error == null)
        and (.choices[0].finish_reason != null)
        and (((.choices[0].message.reasoning // "")
            + (.choices[0].message.content // "")) | length > 0)' "$1" >/dev/null 2>&1
}

run_pair() { # $1 prefix
    local prefix=$1 gate="$EV/$1.go" ready="$EV/$1.ready" p1 p2 rc1=0 rc2=0
    rm -f "$gate" "$ready"
    : > "$ready"
    pair_req() {
        printf 'ready\n' >> "$ready"
        while [ ! -e "$gate" ]; do sleep 0.01; done
        req "$1" 0 "$2"
    }
    pair_req "$P2" "$EV/$prefix-c1.json" & p1=$!
    pair_req "$P3" "$EV/$prefix-c2.json" & p2=$!
    for _ in $(seq 1 500); do [ "$(wc -l < "$ready" 2>/dev/null || echo 0)" -ge 2 ] && break; sleep 0.01; done
    if [ "$(wc -l < "$ready" 2>/dev/null || echo 0)" -lt 2 ]; then
        kill -TERM "$p1" "$p2" 2>/dev/null || true
        wait "$p1" 2>/dev/null || true
        wait "$p2" 2>/dev/null || true
        return 1
    fi
    : > "$gate"
    wait "$p1" || rc1=$?
    wait "$p2" || rc2=$?
    [ "$rc1" = 0 ] && [ "$rc2" = 0 ]
}

P1='"Write a Python function that parses an ISO-8601 timestamp and returns epoch seconds."'
P2='"Describe, in three sentences, why tidal forces lock a moon to its planet."'
P3='"List the shell commands to find the five largest files under /var and archive them."'
P4='"Print exactly 512 comma-separated integers, starting at 1 and ending at 512, without commentary or stopping early."'
P5='"Explain why a sampled speculative session must remain the only non-demotable drafting session."'

echo "== boot 1: dspark route armed =="
boot "MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_SPEC_GATE_LOW=2" \
    "$EV/serve-on.log" || exit 1
i=1
for P in "$P1" "$P2" "$P3"; do
    req "$P" 0 "$EV/on-r$i.json"
    if spec_engaged "$EV/on-r$i.json"; then
        ROUNDS=$(jq -r '.usage.spec.rounds' "$EV/on-r$i.json")
        DRAFTED=$(jq -r '.usage.spec.drafted' "$EV/on-r$i.json")
        ACCEPTED=$(jq -r '.usage.spec.accepted' "$EV/on-r$i.json")
        echo "on-r$i: spec rounds=$ROUNDS drafted=$DRAFTED accepted=$ACCEPTED"
    else
        echo "FAIL: on-r$i did not run speculative rounds and draft tokens"; FAIL=1
    fi
    i=$((i+1))
done
echo "-- sampled rejection-verify engagement --"
req "$P1" 0.7 "$EV/on-sampled.json"
SLEN=$(jq -r '((.choices[0].message.reasoning // "") + (.choices[0].message.content // "")) | length' "$EV/on-sampled.json" 2>/dev/null)
if [ -z "$SLEN" ] || [ "$SLEN" = "null" ] || [ "$SLEN" -eq 0 ]; then
    echo "FAIL: sampled request returned no content"; FAIL=1
elif ! spec_engaged "$EV/on-sampled.json"; then
    echo "FAIL: sampled request silently missed the dspark route"; FAIL=1
else
    echo "sampled: dspark engaged, ${SLEN} chars"
fi
echo "-- concurrent pair (LOW=2 wave admission) --"
run_pair on || { echo "FAIL: synchronized LOW=2 pair did not complete"; FAIL=1; }
for f in on-c1 on-c2; do
    L=$(jq -r '((.choices[0].message.reasoning // "") + (.choices[0].message.content // "")) | length' "$EV/$f.json" 2>/dev/null)
    ERR=$(jq -r '.error // empty' "$EV/$f.json" 2>/dev/null)
    if [ -n "$ERR" ] || [ -z "$L" ] || [ "$L" = "null" ] || [ "$L" -eq 0 ]; then
        echo "FAIL: concurrent $f returned no content"; FAIL=1
    fi
    if ! spec_engaged "$EV/$f.json"; then
        echo "FAIL: concurrent $f ran no rounds/drafts under LOW=2"; FAIL=1
    fi
done
echo "-- sampled-first blocks later greedy LOW widening --"
ACC_BEFORE=$(grep -c "\[dspark-acc\]" "$EV/serve-on.log" 2>/dev/null || true)
ACC_BEFORE=${ACC_BEFORE:-0}
req_n "$P4" 0.7 512 "$EV/on-sampled-first.json" & REQUEST_PID=$!
MIX_READY=0
for _ in $(seq 1 1000); do
    ACC_NOW=$(grep -c "\[dspark-acc\]" "$EV/serve-on.log" 2>/dev/null || true)
    ACC_NOW=${ACC_NOW:-0}
    if [ "$ACC_NOW" -gt "$ACC_BEFORE" ] && kill -0 "$REQUEST_PID" 2>/dev/null; then
        MIX_READY=1
        break
    fi
    kill -0 "$REQUEST_PID" 2>/dev/null || break
    sleep 0.01
done
if [ "$MIX_READY" = 1 ]; then
    req "$P5" 0 "$EV/on-greedy-behind-sampled.json"
else
    echo "FAIL: sampled-first request did not expose a live DFlash round"; FAIL=1
fi
MIX_RC=0
wait "$REQUEST_PID" || MIX_RC=$?
REQUEST_PID=""
if [ "$MIX_RC" != 0 ] || ! valid_completion "$EV/on-sampled-first.json" \
    || ! spec_engaged "$EV/on-sampled-first.json"; then
    echo "FAIL: sampled-first request was not a valid engaged completion"; FAIL=1
fi
if [ "$MIX_READY" = 1 ]; then
    if ! valid_completion "$EV/on-greedy-behind-sampled.json"; then
        echo "FAIL: greedy-behind-sampled request was not a valid completion"; FAIL=1
    elif ! spec_engaged "$EV/on-greedy-behind-sampled.json"; then
        echo "FAIL: greedy request behind a live sampled row did not stack into the LOW band (c4432f4a4 policy)"; FAIL=1
    else
        echo "sampled-first tooth: later greedy request stacked into the LOW band"
    fi
fi
grep -c "\[dspark-acc\]" "$EV/serve-on.log" >/dev/null || { echo "FAIL: no [dspark-acc] in server log"; FAIL=1; }
stop

echo "== boot 1b: LOW=1 negative-control tooth =="
boot "MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_SPEC_GATE_LOW=1" \
    "$EV/serve-low1.log" || exit 1
run_pair low1 || { echo "FAIL: synchronized LOW=1 pair did not complete"; FAIL=1; }
LOW1_SPEC=0
for f in low1-c1 low1-c2; do
    L=$(jq -r '((.choices[0].message.reasoning // "") + (.choices[0].message.content // "")) | length' "$EV/$f.json" 2>/dev/null)
    ERR=$(jq -r '.error // empty' "$EV/$f.json" 2>/dev/null)
    if [ -n "$ERR" ] || [ -z "$L" ] || [ "$L" = "null" ] || [ "$L" -eq 0 ]; then
        echo "FAIL: synchronized LOW=1 $f returned an error or no content"; FAIL=1
    fi
    spec_plain "$EV/$f.json" || LOW1_SPEC=$((LOW1_SPEC+1))
done
if [ "$LOW1_SPEC" = 0 ]; then
    echo "LOW=1 tooth: synchronized c=2 wave admitted zero drafting rows"
else
    echo "FAIL: LOW=1 tooth saw $LOW1_SPEC drafting rows (want 0 for projected wave=2)"; FAIL=1
fi
stop

echo "== boot 2: spec off (plain oracle) =="
boot "MEMRA_SERVE_SPEC=0" "$EV/serve-off.log" || exit 1
i=1
for P in "$P1" "$P2" "$P3"; do
    req "$P" 0 "$EV/off-r$i.json"
    i=$((i+1))
done
stop

echo "== byte identity =="
for i in 1 2 3; do
    A=$(jq -r '(.choices[0].message.reasoning // "") + "\u001f" + (.choices[0].message.content // "")' "$EV/on-r$i.json" | sha256sum | cut -c1-16)
    B=$(jq -r '(.choices[0].message.reasoning // "") + "\u001f" + (.choices[0].message.content // "")' "$EV/off-r$i.json" | sha256sum | cut -c1-16)
    if [ "$A" = "$B" ]; then
        echo "r$i IDENTICAL ($A)"
    else
        echo "FAIL: r$i DIVERGED on=$A off=$B"; FAIL=1
    fi
done
if [ "$FAIL" = "0" ]; then echo "dspark-serve-smoke: ALL GREEN"; else echo "dspark-serve-smoke: FAILED"; exit 1; fi
