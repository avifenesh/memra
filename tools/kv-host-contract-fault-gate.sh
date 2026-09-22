#!/usr/bin/env bash
# kv-host-contract-fault-gate.sh: the contract-routed host-tier D2H (MEMRA_KV_HOST_CONTRACTS=1,
# lane/spill-c-20260919 Option B) and H2D (Option C) must UNWIND a refusal without wedging the tier.
# Six cells, one server boot each, door ON, one one-shot fault each (docs/FLAGS.md
# `MEMRA_KV_HOST_FAULT`). The demote side first, the two unwind paths the PR #599 review found broken:
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
# Option C (day 16) adds the promote side of the same door, two more cells, one boot each:
#   promote-presubmit    MEMRA_KV_HOST_FAULT=contract-promote-presubmit: the H2D producer fence is
#                        refused before any op is submitted. Every fresh destination must release,
#                        every source twin drop, the host entry stay intact, the promote refuse typed
#                        (`promote refused (contracts door): tier H2D producer fence refused: ...`),
#                        the tier stay on, and the NEXT promote complete with an H2D receipt.
#   promote-postpublish  MEMRA_KV_HOST_FAULT=contract-promote-postpublish: a refusal after
#                        `ready_view` published every item. The published ticket must retire against
#                        its consumer fence after its sources retired and be acknowledged, the
#                        promote refuse typed, the tier stay on, and the NEXT promote complete.
#   promote-reject       MEMRA_KV_HOST_FAULT=contract-promote-reject (PR #605 finding 1): the last op of the
#                        batch is mis-sized by one byte and the engine rejects exactly it; the unwind must
#                        recover only the accepted sources and end as a plain refusal, tier on. The refusal
#                        names `1 of N items`; N is the entry's plane count (a property of the artifact: 34
#                        on the 27B, 18 on the 9B) and is read from the server's own r2 D2H receipt for the
#                        same entry (`contracts door D2H receipt: ... items=N`), never from a table here.
#   promote-readyview    MEMRA_KV_HOST_FAULT=contract-promote-readyview (PR #605 finding 2): the first
#                        `ready_view` published the ticket in the engine but the route sees a failure; the
#                        unwind must ask the engine and take the published arm, nothing leaked, tier on.
# Promote cell shape: r1 P_A seeds E_A; r2 P_B seeds E_B, evicts E_A (a clean demote, D2H seq=1);
# r3 P_A again: device miss, host hit, the promote takes the injected refusal and the cold path
# serves (its insert evicts E_B into a clean demote); r4 P_B: host hit, the promote must complete
# (an H2D receipt, then `[prefix-host] promote:`). Four 200s.
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
after() { # $1 literal $2 regex $3 log: the first match of regex $2 comes after the first line containing literal $1
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
a, b, log = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(log, errors="replace").read().splitlines()
ia = next((i for i, l in enumerate(lines) if a in l), None)
ib = next((i for i, l in enumerate(lines) if re.search(b, l)), None)
sys.exit(0 if ia is not None and ib is not None and ib > ia else 1)
PY
}
three_served() { # $1 evidence prefix: r1..r3 each carry a non-empty completion
    python3 - "$1" <<'PY'
import json, sys
p = sys.argv[1]
sys.exit(0 if all(json.load(open(f"{p}-r{i}.json"))["choices"][0]["text"] for i in (1, 2, 3)) else 1)
PY
}
no_extra_refusal() { # $1 log: no host-tier refusal line beyond the injected one
    [ "$(grep -cE '\[prefix-host\] (demote refused|promote refused|REFUSED|.*\(contracts door\): )' "$1")" -eq 0 ]
}
not_leaked() { ! grep -q 'Capacity' "$1" && ! grep -q 'leaked' "$1"; }

cell() { # $1 name $2 fault $3 refused-kind $4 expected next-receipt seq
    local name=$1 fault=$2 kind=$3 seq=$4 log="$EV/$1-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot) =="
    boot "MEMRA_KV_HOST_FAULT=$fault" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_B" "$EV/$name-r2.json"
    req "$P_C" "$EV/$name-r3.json"
    stop
    chk "$name: three completions served" three_served "$EV/$name"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: exactly one typed injected refusal, the $kind" count_eq "demote failed (tier D2H $kind refused: injected failure (MEMRA_KV_HOST_FAULT=$fault)); nothing demoted" "$log" 1
    chk "$name: the next demote completes with a D2H contract receipt after the refusal" after "demote failed (tier D2H $kind refused: injected" "contracts door D2H receipt: ticket issuer=[0-9]+ seq=$seq .* require=ok" "$log"
    chk "$name: the next demote publishes" after "demote failed (tier D2H $kind refused: injected" "\\[prefix-host\\] demote: " "$log"
    chk "$name: the tier never latched off" absent "TIER DISABLED" "$log"
    chk "$name: no entry was dropped as not whole (no quarantine)" absent "no longer whole" "$log"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" not_leaked "$log"
    chk "$name: no host-tier refusal line beyond the injected one" no_extra_refusal "$log"
}

after_any() { # $1 literal $2 regex $3 log: SOME match of regex $2 comes after the first line containing literal $1
    # (the promote cells have receipts before the refusal too, so `after`'s first-match rule would read the r2 receipt)
    python3 - "$1" "$2" "$3" <<'PYEOF'
import re, sys
a, b, log = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(log, errors="replace").read().splitlines()
ia = next((i for i, l in enumerate(lines) if a in l), None)
sys.exit(0 if ia is not None and any(re.search(b, l) for l in lines[ia + 1:]) else 1)
PYEOF
}
one_refusal_only() { # $1 log $2 refusal: the only host-tier refusal line is the injected one
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
log, refusal = sys.argv[1], sys.argv[2]
pat = re.compile(r"\[prefix-host\] (demote refused|promote refused|REFUSED|.*\(contracts door\): |demote failed)")
lines = [l for l in open(log, errors="replace").read().splitlines() if pat.search(l)]
sys.exit(0 if len(lines) == 1 and refusal in lines[0] else 1)
PYEOF
}
four_served() { # $1 evidence prefix: r1..r4 each carry a non-empty completion
    python3 - "$1" <<'PYEOF'
import json, sys
p = sys.argv[1]
sys.exit(0 if all(json.load(open(f"{p}-r{i}.json"))["choices"][0]["text"] for i in (1, 2, 3, 4)) else 1)
PYEOF
}

d2h_receipt_items() { # $1 log: `items=N` of the FIRST D2H receipt (r2's demote of E_A, the entry r3 promotes)
    grep -m1 -oE 'contracts door D2H receipt: ticket issuer=[0-9]+ seq=[0-9]+ .* items=[0-9]+' "$1" \
        | grep -oE 'items=[0-9]+' | cut -d= -f2
}
reject_total_matches_receipt() { # $1 log $2 items: the `1 of M items` M in the injected refusal equals the receipt's N
    local m
    m=$(grep -m1 -oE 'tier H2D batch partially refused: 1 of [0-9]+ items \(injected failure' "$1" \
        | grep -oE 'of [0-9]+ items' | grep -oE '[0-9]+')
    [ -n "$2" ] && [ -n "$m" ] && [ "$m" -eq "$2" ]
}
pcell() { # $1 name $2 fault $3 refused-kind-or-literal: a kind ("producer fence", "publication") names the
          # `tier H2D <kind> refused: injected failure (...)` shape; a value starting with `tier H2D ` is the whole
          # typed reason (the readyview cell carries the engine's own wording plus the injected marker); the
          # literal `partial-reject` builds the reject cell's reason from the server's own D2H receipt item count
    local name=$1 fault=$2 kind=$3 log="$EV/$1-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot, promote side) =="
    boot "MEMRA_KV_HOST_FAULT=$fault" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_B" "$EV/$name-r2.json"
    req "$P_A" "$EV/$name-r3.json"
    req "$P_B" "$EV/$name-r4.json"
    stop
    local reason items=""
    case "$kind" in
        partial-reject)
            items=$(d2h_receipt_items "$log")
            echo "  $name: the entry's plane count from the server's r2 D2H receipt: items=${items:-<none>}"
            chk "$name: the r2 D2H receipt names the entry's item total (the artifact's plane count)" \
                test -n "$items"
            chk "$name: the injected refusal's \`1 of M items\` M equals that receipt's items=N" \
                reject_total_matches_receipt "$log" "$items"
            # FLOOR (revuto on #626): a PARTIAL reject needs at least one accepted plane beside the
            # rejected one; items=1 would satisfy the self-consistency check while making the cell
            # a whole-batch refusal. Every artifact class read here so far carries far more (27B
            # 34 default / 32 plain, 9B 18 / 16).
            chk "$name: the entry carries at least two planes, so the reject is partial (items=N >= 2)" \
                test "${items:-0}" -ge 2
            reason="tier H2D batch partially refused: 1 of ${items:-0} items (injected failure (MEMRA_KV_HOST_FAULT=$fault))"
            ;;
        "tier H2D "*) reason="$kind" ;;
        *) reason="tier H2D $kind refused: injected failure (MEMRA_KV_HOST_FAULT=$fault)" ;;
    esac
    local refusal="promote refused (contracts door): $reason; serving without the host entry"
    chk "$name: four completions served" four_served "$EV/$name"
    chk "$name: door ON with the transfer engine on both sides" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine.*KV plane H2D through the same engine on promote" "$log"
    chk "$name: exactly one typed injected refusal, the $kind" count_eq "$refusal" "$log" 1
    chk "$name: a clean demote with a D2H receipt follows the refusal (the cold path's insert evicted)" after_any "$refusal" "contracts door D2H receipt: ticket issuer=[0-9]+ seq=[0-9]+ .* require=ok" "$log"
    chk "$name: the next promote completes with an H2D contract receipt after the refusal" after_any "$refusal" "contracts door H2D receipt: ticket issuer=[0-9]+ seq=[0-9]+ .* require=ok .* published retired acknowledged" "$log"
    chk "$name: the next promote publishes" after_any "$refusal" "\\[prefix-host\\] promote: " "$log"
    chk "$name: the promote refusal kept the host entry" absent "host entry dropped" "$log"
    chk "$name: the tier never latched off" absent "TIER DISABLED" "$log"
    chk "$name: no entry was dropped as not whole (no quarantine)" absent "no longer whole" "$log"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" not_leaked "$log"
    chk "$name: no host-tier refusal line beyond the injected one" one_refusal_only "$log" "$refusal"
}

cell presubmit contract-presubmit "producer fence" 1
cell postpublish contract-postpublish receipt 2
pcell promote-presubmit contract-promote-presubmit "producer fence"
pcell promote-postpublish contract-promote-postpublish publication
# PR #605 review: a partially accepted batch (one op the engine rejects) and a first ready_view reported failed
# while the engine has published; both must end as plain refusals with the ticket retired and acknowledged.
pcell promote-reject contract-promote-reject partial-reject
pcell promote-readyview contract-promote-readyview "tier H2D destination 0 not publishable: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-readyview)"

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"
else
    echo "KV-HOST-CONTRACT-FAULT GATE: $FAILS FAILURE(S)"
    exit 1
fi
