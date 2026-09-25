#!/usr/bin/env bash
# kv-host-pause-demote-gate.sh: the agent-pause demote (MEMRA_KV_PAUSE_DEMOTE=1) under the host-tier contracts door
# (MEMRA_KV_HOST_CONTRACTS=1) with its demotes off the tick (WP-A day 47, research/spill-a-20260919/DAY47.md design V,
# OWED item 6). A tool conversation (the darklanes pause battery's shape): turn 1 declares two tools under an agent
# system prompt and must end in a tool call (finish_reason tool_calls; otherwise the cell measures nothing and FAILS);
# its retire arms a pause candidate; the sweep fires MEMRA_KV_PAUSE_DEMOTE_MS later; turn 2 appends the assistant turn
# and a fixed tool result. Two boots per shape set: plain (MEMRA_SERVE_SPEC=0: shape 1, the continuation park, then
# shape 2, the resident prefix entry) and default (spec parks are out of the pause's scope: shape 2 only). Cells, one
# server boot each, every turn byte-compared with a door-OFF, pause-OFF reference boot of the same turns:
#   clean    the release lines of the boot's shapes (off the tick), turn 2 a host hit (cached_tokens > 0) or a park.
#   race     MEMRA_KV_HOST_FAULT=d2h-delay (the boot's first demote held 3 s unlanded): turn 2 sent once the pause
#            fired parks on the Demoting entry (the plain boot's is shape 1's snapshot) and promotes after the
#            publication; the plain boot's publication then releases the unused park (DAY47 section 3a).
#   failure  MEMRA_KV_HOST_FAULT=contract-presubmit (the boot's first contract D2H refused before any op). Plain: the
#            park is kept (`host copy did not publish; park kept`). Default: the entry is reinstated (`entry
#            reinstated`). The tier stays on.
#
# usage: kv-host-pause-demote-gate.sh [--external-lock FD] <model.gguf> <server_bin> <evidence_dir>
# env:   MEMRA_PAUSEGATE_BOOTS (default "plain default")  MEMRA_HOSTGATE_CACHE_MB (default 256)
#        MEMRA_HOSTGATE_HOST_MB (default 8192)
# Boots its own servers one cell at a time (flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}). Exit 0 = every assertion
# held. Evidence: <evidence_dir>/<boot>-<cell>-t{1,2}.json + <boot>-<cell>-server.log.
set -euo pipefail
LOCK_FD=""
LOCK_OWNER=internal-canonical
if [[ ${1:-} == --external-lock ]]; then
    [[ ${2:-} =~ ^[0-9]+$ ]] || { echo "REFUSED: inherited lock FD required" >&2; exit 2; }
    LOCK_FD=$2
    LOCK_OWNER=collector
    shift 2
fi
[[ $# == 3 ]] || { echo "REFUSED: expected MODEL BIN EV" >&2; exit 2; }
MODEL=$1
BIN=$2
EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18119}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "kv-host-pause-demote-gate: FAIL: $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
case "$GPU_LOCK" in
    /tmp/memra-5090.lock|/tmp/memra-gpu.lock) ;;
    *) echo "REFUSED: noncanonical GPU lock" >&2; exit 2 ;;
esac
if [[ $LOCK_OWNER == internal-canonical ]]; then
    exec 9>"$GPU_LOCK"
    LOCK_FD=9
    flock -n "$LOCK_FD" || { echo "REFUSED: canonical GPU lock busy" >&2; exit 2; }
fi
LOCK_PROOF=$(python3 "$HERE/tier-lock-proof.py" --fd "$LOCK_FD" --lock "$GPU_LOCK" --owner "$LOCK_OWNER")
mkdir -p "$EV"
printf '%s\n' "$LOCK_PROOF" > "$EV/LOCK.json"
SERVER_PID=""
CACHE_MB=${MEMRA_HOSTGATE_CACHE_MB:-256}
HOST_MB=${MEMRA_HOSTGATE_HOST_MB:-8192}
BOOTS=${MEMRA_PAUSEGATE_BOOTS:-plain default}

boot() { # $1 extra-env-string  $2 log
    memra_port_guard kv-host-pause-demote-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"
        return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" "MEMRA_KV_HOST_MB=$HOST_MB" $1 "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            echo "server died during boot:"
            tail -20 "$2"
            return 1
        }
        sleep 2
    done
    echo "server never became ready"
    return 1
}
stop() {
    [[ -n $SERVER_PID ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop EXIT

turn() { # $1 1|2 $2 out-json [$3 turn-1 json for turn 2]
    python3 - "$PORT" "$1" "$2" "${3:-}" <<'PY'
import json, sys, time, urllib.request
port, which, out, prev = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repository-relative path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"}}, "required": ["command"]}}},
]
SYS = ("You are a coding agent working in a git repository. You have tools. When you need information from the "
       "repository, CALL A TOOL instead of guessing. Do not explain what you are about to do; just call the tool.")
msgs = [{"role": "system", "content": SYS},
        {"role": "user", "content": "Open notes/tide-gauges.md and tell me what its first section says."}]
if which == "2":
    t1 = json.load(open(prev))
    m = t1["choices"][0]["message"]
    a = {"role": "assistant", "content": m.get("content")}
    if m.get("reasoning_content") or m.get("reasoning"):
        a["reasoning_content"] = m.get("reasoning_content") or m.get("reasoning")
    if m.get("tool_calls"):
        a["tool_calls"] = m["tool_calls"]
    msgs.append(a)
    for c in m.get("tool_calls") or []:
        f = c.get("function") or {}
        msgs.append({"role": "tool", "tool_call_id": c.get("id", "call_0"),
                     "content": f"[tool {f.get('name')} result] ok. arguments were: {f.get('arguments')}. Contents: "
                                "the first section lists twelve stations north to south with their datum epochs."})
body = {"model": "gate", "messages": msgs, "tools": TOOLS, "max_tokens": 256, "temperature": 0}
t0 = time.monotonic()
r = urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600)
j = json.load(r)
j["_gate_wall_ms"] = (time.monotonic() - t0) * 1e3
json.dump(j, open(out, "w"), indent=1)
PY
}

FAILS=0
chk() { # NAME CMD...
    local name=$1
    shift
    if "$@"; then echo "  ok: $name"; else
        echo "  FAIL: $name"
        FAILS=$((FAILS + 1))
    fi
}
absent() { ! grep -q "$1" "$2"; }
present() { grep -q "$1" "$2"; }
await_line() { # $1 literal $2 log: bounded 150 x 100 ms (a fixed string: WP-A day 47 section 3a, the first run's
             # `[prefix-host] pause armed` was read as a bracket expression and the check errored)
    for _ in $(seq 1 150); do grep -qF -- "$1" "$2" && return 0; sleep 0.1; done
    return 1
}
ended_in_tool_call() { # $1 json
    python3 - "$1" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
c = j["choices"][0]
sys.exit(0 if c.get("finish_reason") == "tool_calls" and c["message"].get("tool_calls") else 1)
PY
}
cached_positive() { # $1 json
    python3 - "$1" <<'PY'
import json, sys
u = json.load(open(sys.argv[1])).get("usage", {})
sys.exit(0 if ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cached_tokens") or 0) > 0 else 1)
PY
}
same_turn() { # $1 json A $2 json B: the message (content, reasoning, tool calls) byte-equal
    python3 - "$1" "$2" <<'PY'
import json, sys
def key(p):
    m = json.load(open(p))["choices"][0]["message"]
    return json.dumps([m.get("content"), m.get("reasoning_content") or m.get("reasoning"), m.get("tool_calls")],
                      sort_keys=True)
sys.exit(0 if key(sys.argv[1]) == key(sys.argv[2]) else 1)
PY
}
PAUSE="MEMRA_KV_HOST_CONTRACTS=1 MEMRA_KV_PAUSE_DEMOTE=1 MEMRA_KV_PAUSE_DEMOTE_MS=300"

for b in $BOOTS; do
    case $b in
        plain) SPEC="MEMRA_SERVE_SPEC=0" ;;
        default) SPEC="" ;;
        *) echo "REFUSED: boot $b"; exit 2 ;;
    esac
    echo "== $b: the reference boot (door OFF, pause OFF) =="
    boot "$SPEC MEMRA_KV_HOST_CONTRACTS=0" "$EV/$b-ref-server.log"
    turn 1 "$EV/$b-ref-t1.json"
    turn 2 "$EV/$b-ref-t2.json" "$EV/$b-ref-t1.json"
    stop
    chk "$b ref: turn 1 ended in a tool call (the cell measures a pause)" ended_in_tool_call "$EV/$b-ref-t1.json"

    echo "== $b clean =="
    L=$EV/$b-clean-server.log
    boot "$SPEC $PAUSE" "$L"
    turn 1 "$EV/$b-clean-t1.json"
    armed=0; await_line "[prefix-host] pause armed" "$L" || armed=$?
    if [ "$b" = plain ]; then
        rel1=0; await_line "pause demote: plain park released off the tick" "$L" || rel1=$?
        chk "$b clean: shape 1 released off the tick" test "$rel1" -eq 0
    fi
    rel2=0; await_line "pause demote: device prefix entry released off the tick" "$L" || rel2=$?
    turn 2 "$EV/$b-clean-t2.json" "$EV/$b-clean-t1.json"
    stop
    chk "$b clean: the pause armed" test "$armed" -eq 0
    chk "$b clean: shape 2 released off the tick" test "$rel2" -eq 0
    chk "$b clean: no pause demote on the tick" absent "pause demote: device prefix entry released (" "$L"
    chk "$b clean: nothing reinstated or kept" absent "reinstated\|park kept" "$L"
    chk "$b clean: turn 2 hit the kept state (cached_tokens > 0)" cached_positive "$EV/$b-clean-t2.json"
    chk "$b clean: turn 1 byte-equal to the reference" same_turn "$EV/$b-clean-t1.json" "$EV/$b-ref-t1.json"
    chk "$b clean: turn 2 byte-equal to the reference" same_turn "$EV/$b-clean-t2.json" "$EV/$b-ref-t2.json"
    chk "$b clean: the tier never latched off" absent "TIER DISABLED" "$L"

    echo "== $b race (MEMRA_KV_HOST_FAULT=d2h-delay) =="
    L=$EV/$b-race-server.log
    boot "$SPEC $PAUSE MEMRA_KV_HOST_FAULT=d2h-delay" "$L"
    turn 1 "$EV/$b-race-t1.json"
    fired=0; await_line "demote fault armed (MEMRA_KV_HOST_FAULT=d2h-delay)" "$L" || fired=$?
    turn 2 "$EV/$b-race-t2.json" "$EV/$b-race-t1.json"
    # WP-A day 47 section 3a: in both boots turn 2 parks on the Demoting entry (in the plain boot, shape 1's snapshot:
    # the continuation park does not serve turn 2 in this shape) and promotes after the publication; the plain boot's
    # publication then releases the unused park.
    chk "$b race: turn 2 parked on the Demoting entry" present "hit parked on a Demoting entry" "$L"
    chk "$b race: turn 2 promoted after the publication" present "\[prefix-host\] promote: " "$L"
    if [ "$b" = plain ]; then
        rel=0; await_line "pause demote: plain park released off the tick" "$L" || rel=$?
        chk "$b race: the publication released the unused park" test "$rel" -eq 0
    fi
    stop
    chk "$b race: the held demote was the pause's" test "$fired" -eq 0
    chk "$b race: turn 2 byte-equal to the reference" same_turn "$EV/$b-race-t2.json" "$EV/$b-ref-t2.json"
    chk "$b race: the tier never latched off" absent "TIER DISABLED" "$L"

    echo "== $b failure (MEMRA_KV_HOST_FAULT=contract-presubmit) =="
    L=$EV/$b-failure-server.log
    boot "$SPEC $PAUSE MEMRA_KV_HOST_FAULT=contract-presubmit" "$L"
    turn 1 "$EV/$b-failure-t1.json"
    if [ "$b" = plain ]; then
        kept=0; await_line "pause demote: host copy did not publish; park kept" "$L" || kept=$?
        chk "$b failure: shape 1's park kept" test "$kept" -eq 0
    else
        back=0; await_line "pause demote failed: device prefix entry reinstated" "$L" || back=$?
        chk "$b failure: shape 2's entry reinstated" test "$back" -eq 0
    fi
    turn 2 "$EV/$b-failure-t2.json" "$EV/$b-failure-t1.json"
    stop
    chk "$b failure: the refusal is the injected one" present "injected failure (MEMRA_KV_HOST_FAULT=contract-presubmit)" "$L"
    chk "$b failure: turn 2 served from the device (cached_tokens > 0)" cached_positive "$EV/$b-failure-t2.json"
    chk "$b failure: turn 2 byte-equal to the reference" same_turn "$EV/$b-failure-t2.json" "$EV/$b-ref-t2.json"
    chk "$b failure: the tier never latched off" absent "TIER DISABLED" "$L"
done

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-PAUSE-DEMOTE GATE: ALL GREEN"
    exit 0
fi
echo "KV-HOST-PAUSE-DEMOTE GATE: $FAILS FAILURE(S)"
exit 1
