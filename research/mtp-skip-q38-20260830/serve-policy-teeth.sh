#!/usr/bin/env bash
# MEMRA_MTP_SKIP x serve policy teeth (worker-level; needs a full model load):
#  1. skip + explicit MEMRA_SERVE_SPEC=1 + NO dspark drafter -> boot FATAL, cause quoted.
#  2. skip + default spec + NO dspark -> boots, announces PLAIN loudly, and a greedy request
#     serves with .usage.spec == null (the stub draft head must NOT satisfy mtp_spec_capable).
# usage: serve-policy-teeth.sh <memra-server-bin> <q38.gguf> <ranks.txt> <evidence_dir>
set -uo pipefail
BIN=$1; MODEL=$2; RANKS=$3; EV=$4
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18133}
mkdir -p "$EV"
FAIL=0

echo "== tooth 1: skip + MEMRA_SERVE_SPEC=1, no dspark -> FATAL =="
LOG="$EV/spec-explicit-fatal.log"
timeout 900 flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
    MEMRA_MTP_SKIP=1 "MEMRA_FRSPEC_TRIM=$RANKS" MEMRA_SERVE_SPEC=1 \
    "$BIN" >"$LOG" 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
    echo "FAIL: server exited 0 under an unhonorable explicit spec request"; FAIL=1
elif grep -qF "cannot be honored" "$LOG"; then
    echo "PASS: exit=$RC, cause quoted:"
    grep -F "cannot be honored" "$LOG" | head -1 | sed 's/^/    /'
else
    echo "FAIL: exit=$RC but the refusal cause is missing"; tail -5 "$LOG" | sed 's/^/    /'; FAIL=1
fi

echo "== tooth 2: skip + default spec, no dspark -> loud PLAIN + plain-served request =="
LOG="$EV/plain-announce.log"
SERVER_PID=""
stop() {
    [ -n "$SERVER_PID" ] || return 0
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
setsid flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
    MEMRA_MTP_SKIP=1 "MEMRA_FRSPEC_TRIM=$RANKS" \
    "$BIN" >"$LOG" 2>&1 &
SERVER_PID=$!
UP=0
for _ in $(seq 1 300); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { UP=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
done
if [ "$UP" != 1 ]; then
    echo "FAIL: skip + default-spec boot did not come up"; tail -5 "$LOG" | sed 's/^/    /'; FAIL=1
else
    if grep -qF "serves PLAIN decode" "$LOG"; then
        echo "PASS: loud PLAIN line present:"
        grep -F "[mtp-skip] gate:" "$LOG" | head -1 | sed 's/^/    /'
    else
        echo "FAIL: no loud PLAIN line in the boot log"; FAIL=1
    fi
    R="$EV/plain-req.json"
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d '{"model":"gate","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"Describe, in three sentences, why tidal forces lock a moon to its planet."}]}' \
        >"$R"
    if jq -e '(.error == null) and (.usage.spec == null) and (((.choices[0].message.reasoning // "") + (.choices[0].message.content // "")) | length > 0)' "$R" >/dev/null 2>&1; then
        echo "PASS: greedy request served PLAIN (.usage.spec == null) despite the loaded stub"
    else
        echo "FAIL: request errored or engaged spec on a skip+no-dspark boot"; FAIL=1
        jq -c '{error: .error, spec: .usage.spec}' "$R" 2>/dev/null | sed 's/^/    /'
    fi
fi
stop

if [ "$FAIL" = 0 ]; then echo "serve-policy-teeth: ALL GREEN"; else echo "serve-policy-teeth: FAILED"; exit 1; fi
