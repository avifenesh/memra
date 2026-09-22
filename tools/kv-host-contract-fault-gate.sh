#!/usr/bin/env bash
# kv-host-contract-fault-gate.sh: the contract-routed host-tier D2H (MEMRA_KV_HOST_CONTRACTS=1,
# lane/spill-c-20260919 Option B) and H2D (Option C) must UNWIND a refusal without wedging the tier.
# Cells, one server boot each, door ON, one one-shot fault each (docs/FLAGS.md
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
# Day 22 (WP-A, memra#536 Move 2 slice 3) adds the D2D receipt's red arm, two cells, one boot each, plain arm:
#   d2d-capture   MEMRA_KV_HOST_FAULT=d2d-delay-capture: the first capture's copy is delayed and its destination
#                 digest read early; the receipt must name two different digests and refuse (require=Corrupt),
#                 nothing publishes, the capture route and the tier latch, r2 is served by the tick program with
#                 r1's bytes.
#   d2d-restore   MEMRA_KV_HOST_FAULT=d2d-delay-restore: r1 seeds through the route (its receipt accepted), r2's
#                 whole-entry hit restores through the route with the fault; the receipt refuses, nothing is primed
#                 on the destination, the cache drops and the pin is released, the route and the tier latch, the
#                 tick program restores r2 with r1's bytes.
#
# WP-A day 28 (memra#536 Move 1 owed item 2, lead ruling 39, research/spill-a-20260919/DAY28.md): the bundle checksum
# of the image's heap payloads runs on one long-lived helper thread per tier context; the demote stays `Demoting`
# (`Hashing`) until the digests land. Two red arms, one boot each, the demote cells' shape (r1 P_A seeds E_A; r2 P_B
# evicts E_A, whose demote copies, completes, hands its heap payloads to the helper and takes the fault; r3 P_C evicts E_B):
#   hash-helper-gone   MEMRA_KV_HOST_FAULT=hash-helper-gone: the helper exits on its first job; the next tick-top poll finds
#                      the reply channel closed; typed refusal, nothing published, the tier latches and joins the helper;
#                      r3's eviction finds the tier off and demotes nothing (no refusal line of its own).
#   hash-never-lands   MEMRA_KV_HOST_FAULT=hash-never-lands: the helper hashes its first job and discards the reply; r3's
#                      eviction meets the Hashing entry through the Block wait ("a second demote") and rides out the 10 s
#                      deadline; typed refusal, nothing published, the tier latches and joins the helper; then r3's own
#                      demote is refused typed (`the tier latched off while settling the pending demote`), exactly once.
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
receipt_seq_accounts() { # $1 refusal literal $2 expected D2H seq without captures $3 log
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
a, expected, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(log, errors="replace").read().splitlines()
ia = next((i for i, l in enumerate(lines) if a in l), None)
if ia is None:
    print("no refusal line"); sys.exit(1)
rx = re.compile(r"contracts door D2H receipt: ticket issuer=[0-9]+ seq=([0-9]+) .* require=ok")
ib = next(((i, m) for i, l in enumerate(lines[ia:], start=ia) if (m := rx.search(l))), None)
if ib is None:
    print("no D2H receipt after the refusal"); sys.exit(1)
i, m = ib
seq = int(m.group(1))
# The sequence is consumed at SUBMISSION. Count the capture tickets submitted before this demote's
# own submission line (Move 1 prints it); a capture submitted between the demote's submission and
# its receipt does not precede it. Trees without the submission line count up to the receipt.
rs = re.compile(r"demote submitted off the tick: .* ticket seq=([0-9]+)")
sub = next(((j, n) for j, l in enumerate(lines[ia:i], start=ia) if (n := rs.search(l))), None)
if sub is not None:
    j, n = sub
    if int(n.group(1)) != seq:
        print(f"the receipt seq={seq} is not the submitted demote's seq={n.group(1)}"); sys.exit(1)
    cut = j
else:
    cut = i
captures = sum(1 for l in lines[:cut] if "capture submitted off the tick" in l)
want = expected + captures
print(f"receipt seq={seq} expected {expected} + {captures} capture ticket(s) submitted before it = {want}")
sys.exit(0 if seq == want else 1)
PY
}

await_hashes() { # $1 log: every `handed to the hash helper` hand-off of the boot has resolved (its digests landed or a
    # typed `demote failed (tier hash` line) before the boot stops. WP-A day 29 (ruling 40): r3 is the boot's LAST
    # request, its eviction's demote hands its heap payloads to the hash helper at r3's boundary, and a `stop` inside the
    # hash (day 28: 73 ms on the target card) read a log that ended at `demote copy complete`, so `the next demote
    # publishes` failed on a publication that had not happened yet. Bounded: 150 x 100 ms = 15 s, above the helper's
    # 10 s deadline, so a hand-off that never lands is read as its typed refusal, not as a timeout here.
    for _ in $(seq 1 150); do
        local handed landed failed
        # `grep -c` exits 1 on a zero count and this gate runs `set -e`: the `|| true` keeps the count.
        handed=$(grep -c "handed to the hash helper" "$1" || true)
        landed=$(grep -c "demote digests landed off the tick" "$1" || true)
        failed=$(grep -c "demote failed (tier hash" "$1" || true)
        [ "$handed" -le $((landed + failed)) ] && return 0
        sleep 0.1
    done
    return 1
}
cell() { # $1 name $2 fault $3 refused-kind $4 expected next-receipt seq
    local name=$1 fault=$2 kind=$3 seq=$4 log="$EV/$1-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot) =="
    boot "MEMRA_KV_HOST_FAULT=$fault" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_B" "$EV/$name-r2.json"
    req "$P_C" "$EV/$name-r3.json"
    # WP-A day 29 (ruling 40): the demote cells await the boot's last publication before `stop`.
    await_hashes "$log"; local awaited=$?
    stop
    chk "$name: the boot's last hand-off landed before stop (bounded 15 s wait)" test "$awaited" -eq 0
    chk "$name: three completions served" three_served "$EV/$name"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: exactly one typed injected refusal, the $kind" count_eq "demote failed (tier D2H $kind refused: injected failure (MEMRA_KV_HOST_FAULT=$fault)); nothing demoted" "$log" 1
    chk "$name: the next demote completes with a D2H contract receipt after the refusal" after "demote failed (tier D2H $kind refused: injected" "contracts door D2H receipt: ticket issuer=[0-9]+ seq=[0-9]+ .* require=ok" "$log"
    # The ticket accounting the literal seq used to carry (spill-c day 24, slice 1 of Move 2): a
    # refused presubmit consumed no sequence (the next D2H is 1 past the tickets before it), a
    # refused postpublish consumed one (2 past). Since Move 2 slice 1 the plain arm's seeds submit
    # CAPTURE tickets on the same issuer, each consuming a sequence, so the receipt's seq is the
    # expected D2H count plus the capture tickets submitted before it, never a literal.
    chk "$name: the receipt's seq is $seq plus the capture tickets submitted before it" receipt_seq_accounts "demote failed (tier D2H $kind refused: injected" "$seq" "$log"
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

two_served() { # $1 evidence prefix: r1 and r2 each carry a non-empty completion
    python3 - "$1" <<'PYEOF'
import json, sys
p = sys.argv[1]
sys.exit(0 if all(json.load(open(f"{p}-r{i}.json"))["choices"][0]["text"] for i in (1, 2)) else 1)
PYEOF
}
texts_equal() { # $1 evidence prefix: r1's text is byte-equal to r2's (the tick program served both)
    python3 - "$1" <<'PYEOF'
import json, sys
p = sys.argv[1]
a = json.load(open(f"{p}-r1.json"))["choices"][0]["text"]
b = json.load(open(f"{p}-r2.json"))["choices"][0]["text"]
sys.exit(0 if a == b else 1)
PYEOF
}
receipt_digests_differ() { # $1 log $2 class: the ONE `require=Corrupt` receipt line of $2 names two DIFFERENT digests
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
log, cls = sys.argv[1], sys.argv[2]
rx = re.compile(rf"contracts door D2D {cls} receipt: .* source_digests_sha256=([0-9a-f]+) destination_digests_sha256=([0-9a-f]+) require=Corrupt")
hits = [m for l in open(log, errors="replace") if (m := rx.search(l))]
if len(hits) != 1:
    print(f"{len(hits)} refused {cls} receipt line(s), expected 1"); sys.exit(1)
src, dst = hits[0].group(1), hits[0].group(2)
print(f"{cls} receipt refused: source_digests_sha256={src[:16]}.. destination_digests_sha256={dst[:16]}..")
sys.exit(0 if src != dst else 1)
PYEOF
}
# WP-A day 22 (memra#536 Move 2 slice 3, research/spill-a-20260919/DAY22.md): the D2D receipt's red arm. The
# plain arm (MEMRA_SERVE_SPEC=0) so the seeds capture through the copy-stream route (the spec arm's seeds publish
# through prefix_insert_from_spec_boundary on the tick and never reach the receipt: a cell that took the fault
# nowhere must fail loudly, which the `exactly one refused receipt` clause does).
dcell_capture() { # MEMRA_KV_HOST_FAULT=d2d-delay-capture: r1 P_A (the capture's early reader reads the fresh plane;
                  # the receipt refuses; nothing published; tier and route latch), r2 P_A (cold; on-tick capture)
    local name=d2d-capture fault=d2d-delay-capture log="$EV/d2d-capture-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot, the D2D capture receipt's red arm) =="
    boot "MEMRA_KV_HOST_FAULT=$fault MEMRA_SERVE_SPEC=0" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_A" "$EV/$name-r2.json"
    stop
    chk "$name: two completions served" two_served "$EV/$name"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: the fault was armed on the first capture" count_eq "capture fault armed (MEMRA_KV_HOST_FAULT=$fault)" "$log" 1
    chk "$name: exactly one refused D2D capture receipt naming two different digests" receipt_digests_differ "$log" capture
    chk "$name: no capture receipt was accepted (the route latched on the first)" absent "D2D capture receipt: .* require=ok" "$log"
    chk "$name: the capture route latched typed on the mismatch" count_eq "CAPTURE OFF-TICK DISABLED: D2D capture receipt mismatch" "$log" 1
    chk "$name: the tier latched off" count_eq "TIER DISABLED" "$log" 1
    chk "$name: nothing was published off the tick" absent "capture published off the tick" "$log"
    chk "$name: nothing was hit (no entry existed before r2)" absent "\[prefix-cache\] hit: " "$log"
    chk "$name: r2's capture ran on the tick after the latch (an insert follows the refusal)" after "CAPTURE OFF-TICK DISABLED: D2D capture receipt mismatch" "\[prefix-cache\] insert \(seed\)" "$log"
    chk "$name: r1's text equals r2's (the tick program served both)" texts_equal "$EV/$name"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" not_leaked "$log"
}
dcell_restore() { # MEMRA_KV_HOST_FAULT=d2d-delay-restore: r1 P_A (a clean capture with its receipt, published), r2 P_A
                  # (a whole-entry hit; the restore's early reader reads the fresh cache; the receipt refuses; nothing
                  # primed on it; the cache drops, the pin is released; tier and route latch; the tick program restores)
    local name=d2d-restore fault=d2d-delay-restore log="$EV/d2d-restore-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot, the D2D restore receipt's red arm) =="
    boot "MEMRA_KV_HOST_FAULT=$fault MEMRA_SERVE_SPEC=0" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_A" "$EV/$name-r2.json"
    stop
    chk "$name: two completions served" two_served "$EV/$name"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: r1's capture receipt was accepted (the green arm, live)" count_eq "D2D capture receipt: .* require=ok" "$log" 1
    chk "$name: r1's capture published off the tick" count_eq "capture published off the tick" "$log" 1
    chk "$name: the fault was armed on the first restore" count_eq "restore fault armed (MEMRA_KV_HOST_FAULT=$fault)" "$log" 1
    chk "$name: exactly one restore was submitted off the tick" count_eq "restore submitted off the tick" "$log" 1
    chk "$name: exactly one refused D2D restore receipt naming two different digests" receipt_digests_differ "$log" restore
    chk "$name: no restore receipt was accepted" absent "D2D restore receipt: .* require=ok" "$log"
    chk "$name: the restore route latched typed on the mismatch" count_eq "RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch" "$log" 1
    chk "$name: the tier latched off" count_eq "TIER DISABLED" "$log" 1
    chk "$name: nothing landed off the tick (nothing primed on the refused cache)" absent "restore landed off the tick" "$log"
    chk "$name: the tick program restored r2 after the refusal" after "RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch" "\[prefix-cache\] hit: " "$log"
    chk "$name: r1's text equals r2's" texts_equal "$EV/$name"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" not_leaked "$log"
}

cell presubmit contract-presubmit "producer fence" 1
cell postpublish contract-postpublish receipt 2
pcell promote-presubmit contract-promote-presubmit "producer fence"
pcell promote-postpublish contract-promote-postpublish publication
# PR #605 review: a partially accepted batch (one op the engine rejects) and a first ready_view reported failed
# while the engine has published; both must end as plain refusals with the ticket retired and acknowledged.
pcell promote-reject contract-promote-reject partial-reject
pcell promote-readyview contract-promote-readyview "tier H2D destination 0 not publishable: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-readyview)"
refusal_names_handoff_ticket() { # $1 log $2 refusal kind: the refusal's `ticket seq=N` is the hand-off line's
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
log, kind = sys.argv[1], sys.argv[2]
lines = open(log, errors="replace").read().splitlines()
hand = [m.group(1) for l in lines if "handed to the hash helper" in l and (m := re.search(r"ticket seq=(\d+)", l))]
ref = [m.group(1) for l in lines if f"demote failed (tier hash {kind}" in l and (m := re.search(r"ticket seq=(\d+)", l))]
print(f"hand-off ticket(s) {hand}, refusal ticket(s) {ref}")
sys.exit(0 if len(hand) == 1 and ref == hand else 1)
PYEOF
}
only_these_refusals() { # $1 log $2.. literals: every host-tier refusal line carries one of the literals
    python3 - "$@" <<'PYEOF'
import re, sys
log, allowed = sys.argv[1], sys.argv[2:]
pat = re.compile(r"\[prefix-host\] (demote refused|promote refused|REFUSED|.*\(contracts door\): |demote failed)")
bad = [l for l in open(log, errors="replace").read().splitlines() if pat.search(l) and not any(a in l for a in allowed)]
for l in bad:
    print("unexpected:", l[:200])
sys.exit(0 if not bad else 1)
PYEOF
}
hcell() { # $1 name $2 fault $3 refusal kind (`helper gone` | `digests never landed`) $4 r3's latched-off refusal count (0|1)
    local name=$1 fault=$2 kind=$3 latched_refusals=$4 log="$EV/$1-server.log"
    echo "== cell $name: MEMRA_KV_HOST_FAULT=$fault (one-shot, the hash helper's red arm) =="
    boot "MEMRA_KV_HOST_FAULT=$fault" "$log"
    req "$P_A" "$EV/$name-r1.json"
    req "$P_B" "$EV/$name-r2.json"
    req "$P_C" "$EV/$name-r3.json"
    stop
    chk "$name: three completions served (the tick program serves)" three_served "$EV/$name"
    chk "$name: door ON with the transfer engine" grep -q "contracts door ON (MEMRA_KV_HOST_CONTRACTS=1).*KV plane D2H through the transfer engine" "$log"
    chk "$name: exactly one copy completed and handed its heap payloads to the hash helper" count_eq "handed to the hash helper" "$log" 1
    chk "$name: exactly one typed injected refusal, $kind" count_eq "demote failed (tier hash $kind" "$log" 1
    chk "$name: the refusal names the hand-off's ticket" refusal_names_handoff_ticket "$log" "$kind"
    chk "$name: the refusal says nothing published and the tier latches" count_eq "; nothing published; the tier latches off" "$log" 1
    chk "$name: the tier latched off exactly once" count_eq "TIER DISABLED" "$log" 1
    chk "$name: the hash helper joined at the latch" count_eq "hash helper joined (the tier latched off)" "$log" 1
    chk "$name: nothing published in the boot (no demote line)" absent "\[prefix-host\] demote: " "$log"
    chk "$name: no digests landed" absent "demote digests landed" "$log"
    chk "$name: no entry was dropped as not whole (no quarantine)" absent "no longer whole" "$log"
    chk "$name: no ticket leaked (no Capacity refusal, no leaked wording)" not_leaked "$log"
    chk "$name: r3's own demote met the latched tier as pre-registered ($latched_refusals refusal line(s))" count_eq "demote refused: the tier latched off while settling the pending demote" "$log" "$latched_refusals"
    chk "$name: no host-tier refusal line beyond the injected one and r3's latched-off refusal" only_these_refusals "$log" "demote failed (tier hash $kind" "demote refused: the tier latched off while settling the pending demote"
}

# WP-A day 22: the D2D receipt's red arm, one cell per class.
dcell_capture
dcell_restore
# WP-A day 28: the hash helper's red arms, one cell each.
hcell hash-helper-gone hash-helper-gone "helper gone" 0
hcell hash-never-lands hash-never-lands "digests never landed" 1

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"
else
    echo "KV-HOST-CONTRACT-FAULT GATE: $FAILS FAILURE(S)"
    exit 1
fi
