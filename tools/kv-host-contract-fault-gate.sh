#!/usr/bin/env bash
# kv-host-contract-fault-gate.sh: the contract-routed host-tier D2H (MEMRA_KV_HOST_CONTRACTS=1,
# lane/spill-c-20260919 Option B) must UNWIND a refusal without wedging the tier. Two cells, one
# server boot each, door ON, one one-shot fault each (docs/FLAGS.md `MEMRA_KV_HOST_FAULT`), the two
# unwind paths the PR #599 review found broken:
#   presubmit    MEMRA_KV_HOST_FAULT=contract-presubmit: the producer fence is refused before any op
#                is submitted. Every registered plane must come back to its slot, the demote fails
#                typed, no ticket exists (the next receipt is `seq=1`), the tier stays on, and the
#                NEXT demote completes with a D2H contract receipt.
#   postpublish  MEMRA_KV_HOST_FAULT=contract-postpublish: the receipt check refuses after every
#                destination was taken. The published ticket must retire against its consumer fence
#                and be acknowledged (the ledger's in-flight dimension is exactly one batch, so a
#                leaked ticket refuses the next demote with `Capacity`), the demote fails typed, the
#                tier stays on, and the NEXT demote completes with a receipt whose ticket is `seq=2`.
# Cell shape (device budget holds ONE seed entry): r1 P_A seeds E_A; r2 P_B seeds E_B, the byte
# budget evicts E_A, whose demote takes the injected refusal; r3 P_C seeds E_C, evicts E_B, whose
# demote must complete. Three 200s, and the server log is the receipt.
#
# usage: kv-host-contract-fault-gate.sh [--external-lock FD] <model.gguf> <server_bin> <evidence_dir>
# env:   MEMRA_HOSTGATE_CACHE_MB (default 256)  device prefix budget; must hold ONE seed entry but not two
#        MEMRA_HOSTGATE_HOST_MB  (default 8192) host tier budget
# Boots its own servers one cell at a time (flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}).
# Exit 0 = every assertion held. Evidence: <evidence_dir>/<cell>-r{1..3}.json + <cell>-server.log.
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
PORT=${MEMRA_GATE_PORT:-18117}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "kv-host-contract-fault-gate: FAIL: $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
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

boot() { # $1 extra-env-string  $2 log
    memra_port_guard kv-host-contract-fault-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"
        return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" "MEMRA_KV_HOST_MB=$HOST_MB" \
        MEMRA_KV_HOST_CONTRACTS=1 $1 "$BIN" >"$2" 2>&1 &
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

# The host-spill gates' two long prompts plus a third of the same shape.
P_A="You are indexing the survey logs of a coastal tide-gauge network. For each of the twelve \
stations, ordered north to south, report the gauge type, the datum epoch, the sampling \
interval in minutes, the last calibration date, the responsible technician role, and the \
anomaly that would force an out-of-cycle calibration. Be systematic and terse; do not skip \
a station. After the twelve stations, add a short paragraph on network-wide drift checks."
P_B="Draft the commissioning checklist for a small hydroelectric turbine hall. Cover, in \
order: penstock inspection, wicket-gate travel, governor response, generator insulation, \
thrust-bearing temperature rise, cooling-water flow, overspeed trip, and grid-synchronization \
tests. For each item name the instrument used, the acceptance threshold, the sign-off role, \
and the failure symptom that would halt commissioning. Be systematic and terse throughout."
P_C="Write the acceptance protocol for a district heating substation. Cover, in order: \
primary-side pressure test, heat exchanger approach temperature, differential pressure \
controller stroke, secondary pump duty and standby changeover, expansion vessel precharge, \
safety valve lift, metering accuracy against the reference meter, and controller failsafe on \
sensor loss. For each item name the instrument, the acceptance threshold, the sign-off role, \
and the symptom that would halt acceptance. Be systematic and terse throughout."

req() { # $1 prompt $2 out-json
    python3 - "$PORT" "$1" "$2" <<'PY'
import json, sys, urllib.request
port, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "gate", "prompt": prompt, "max_tokens": 48, "temperature": 0}
r = urllib.request.urlopen(
    urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    ),
    timeout=600,
)
json.dump(json.load(r), open(out, "w"), indent=1)
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
count_eq() { [ "$(grep -c "$1" "$2")" -eq "$3" ]; }
after() { # $1 first-pattern $2 second-pattern $3 log: the first match of $2 comes after the first match of $1
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
a, b, log = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(log, errors="replace").read().splitlines()
ia = next((i for i, l in enumerate(lines) if re.search(a, l)), None)
ib = next((i for i, l in enumerate(lines) if re.search(b, l)), None)
sys.exit(0 if ia is not None and ib is not None and ib > ia else 1)
PY
}
has_choice() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$1'))['choices'][0]['text'] else 1)"; }

cell() { # $1 name $2 fault $3 refused-kind $4 expected next-receipt seq
    local name=$1 fault=$2 kind=$3 seq=$4 log="$EV/$1-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot) =="
    boot "MEMRA_KV_HOST_FAULT=$fault" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_B" "$EV/$name-r2.json"
    req "$P_C" "$EV/$name-r3.json"
    stop
    chk "$name: three completions served" bash -c "has_choice '$EV/$name-r1.json' && has_choice '$EV/$name-r2.json' && has_choice '$EV/$name-r3.json'"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: exactly one typed injected refusal, the $kind" count_eq "demote failed (tier D2H $kind refused: injected failure (MEMRA_KV_HOST_FAULT=$fault)); nothing demoted" "$log" 1
    chk "$name: the next demote completes with a D2H contract receipt after the refusal" after "demote failed (tier D2H $kind refused: injected" "contracts door D2H receipt: ticket issuer=[0-9]+ seq=$seq .* require=ok" "$log"
    chk "$name: the next demote publishes" after "demote failed (tier D2H $kind refused: injected" "\\[prefix-host\\] demote: " "$log"
    chk "$name: the tier never latched off" absent "TIER DISABLED" "$log"
    chk "$name: no entry was dropped as not whole (no quarantine)" absent "no longer whole" "$log"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" bash -c "! grep -q 'Capacity' '$log' && ! grep -q 'leaked' '$log'"
    chk "$name: no host-tier refusal line beyond the injected one" bash -c "[ \"\$(grep -cE '\\[prefix-host\\] (demote refused|promote refused|REFUSED|.*\\(contracts door\\): )' '$log')\" -eq 0 ]"
}

cell presubmit contract-presubmit "producer fence" 1
cell postpublish contract-postpublish receipt 2

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"
else
    echo "KV-HOST-CONTRACT-FAULT GATE: $FAILS FAILURE(S)"
    exit 1
fi
