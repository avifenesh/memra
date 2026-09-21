#!/usr/bin/env bash
# kv-host-tenant-reclaim-gate.sh: at the host tier's PER-TENANT SHARE CAP a demotion of one
# tenant must evict that tenant's OWN unleased LRU host entry and then demote (memra#384),
# instead of evaporating while the pool still has free space. Serving-shape twin of the
# `host_cache_tenant_share_reclaim_*` unit cells in crates/memra-server/src/worker.rs. One
# binary is booted per invocation and the expectation names the arm: `base` (the pre-#384
# tier: the evaporation line, nothing of the tenant's own evicted) or `fix` (the reclaim
# eviction followed by the demote). The cross-arm comparison (equal `[prefix-host] demote:`
# bytes for the same prompts, equal texts, beta's row equal) is the replay verifier's
# (research/spill-a-20260919/verify-day12.py); this script asserts one arm at a time.
#
# Cell design (two keyring tenants, acme and beta, one PC-ISO salt each; door OFF, the
# artifact's default spec environment):
#   MEMRA_KV_HOST_MB=$HOST_MB (1024)         the pool. At most four ~161 MB entries are ever
#                                            resident, so the pool is never full: the "arena
#                                            still has free space" condition of #384 holds
#                                            throughout (asserted: no `evict (LRU)` line).
#   MEMRA_KV_HOST_TENANT_PCT=$PCT (38)       one tenant's share = 408 MB at 1024 MiB: holds TWO
#                                            ~161 MB entries, not three.
#   MEMRA_PREFIX_CACHE_MB=$CACHE_MB (384)    device budget: TWO entries resident, so every new
#                                            seed evicts the older one into the host tier, and a
#                                            promoted entry fits beside one protected entry (at 256
#                                            the device refused the promote insert: `skip pinned
#                                            host-promote insert ... would evict protected bytes`,
#                                            a device-tier rule, not the host tier's; asserted absent).
#   MEMRA_METRICS_TOKEN                      operator scrape token: under a keyring a tenant bearer
#                                            gets tenant-scope metrics without the process-wide
#                                            prefix_host_* counters this gate reads.
#   r1 beta  P_D        device {E_D}
#   r2 acme  P_A        device {E_D E_A}
#   r3 acme  P_B        evicts E_D -> demote (beta row: E_D)
#   r4 acme  P_C        evicts E_A -> demote (acme row: E_A)
#   r5 acme  P_E        evicts E_B -> demote (acme row: E_A E_B, at its share)
#   r6 acme  P_F        evicts E_C -> acme's THIRD demotion hits the cap with the pool two-thirds
#                       empty:
#                         base: `demote evaporated at the tenant share cap before the D2H copy`
#                         fix:  `evict (tenant share)` of acme's oldest (E_A), then `demote:` E_C
#   r7 beta  P_D + EXT  beta's E_D promotes in BOTH arms (cached_tokens == r1 prompt_tokens):
#                       acme's reclaim never touched beta's row. The promote insert evicts E_E and
#                       the boundary insert evicts E_F: two more cap events (base evaporates both,
#                       fix reclaims E_B then E_C and demotes both).
#   r8 acme  P_F + EXT  fix: E_F promotes (cached_tokens == r6 prompt_tokens), the reclaimed
#                       admission is a real resident entry; base: cold (cached_tokens == 0), the
#                       cost #384 exists to remove. Both arms 200.
#   /metrics            base: prefix_host_tenant_rejects >= 1 and no reclaims; fix: rejects == 0,
#                       prefix_host_tenant_reclaims >= 1.
#
# usage: kv-host-tenant-reclaim-gate.sh [--external-lock FD] <base|fix> <model.gguf> <server_bin> <evidence_dir>
# env:   MEMRA_TENANTGATE_CACHE_MB (384), MEMRA_TENANTGATE_HOST_MB (1024), MEMRA_TENANTGATE_PCT (38),
#        MEMRA_GATE_PORT (18129), MEMRA_GPU_LOCK (/tmp/memra-5090.lock or /tmp/memra-gpu.lock).
# Exit 0 = every assertion for the named arm held. Evidence: <evidence_dir>/r{1..8}.json,
# server.log, metrics.json, SUMMARY.json, LOCK.json.
set -euo pipefail
LOCK_FD=""
LOCK_OWNER=internal-canonical
if [[ ${1:-} == --external-lock ]]; then
    [[ ${2:-} =~ ^[0-9]+$ ]] || { echo "REFUSED: inherited lock FD required" >&2; exit 2; }
    LOCK_FD=$2
    LOCK_OWNER=collector
    shift 2
fi
[[ $# == 4 ]] || { echo "REFUSED: expected ARM MODEL BIN EV" >&2; exit 2; }
ARM=$1
MODEL=$2
BIN=$3
EV=$4
case "$ARM" in base|fix) ;; *) echo "REFUSED: ARM must be base or fix" >&2; exit 2 ;; esac
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18129}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "kv-host-tenant-reclaim-gate: FAIL: $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
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
CACHE_MB=${MEMRA_TENANTGATE_CACHE_MB:-384}
HOST_MB=${MEMRA_TENANTGATE_HOST_MB:-1024}
PCT=${MEMRA_TENANTGATE_PCT:-38}
# Two keyring tenants in the inline form (`tenant:sha256hex`): the plaintext keys are gate
# fixtures, not secrets, and the server never sees them except as bearer tokens.
KEY_ACME=tenant-reclaim-gate-acme
KEY_BETA=tenant-reclaim-gate-beta
METRICS_TOKEN=tenant-reclaim-gate-metrics
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
KEYRING="acme:$(sha "$KEY_ACME"),beta:$(sha "$KEY_BETA")"

boot() { # $1 log
    memra_port_guard kv-host-tenant-reclaim-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving: refusing to boot over it"
        return 1
    fi
    env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_API_KEYS=$KEYRING" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" "MEMRA_KV_HOST_MB=$HOST_MB" \
        "MEMRA_KV_HOST_TENANT_PCT=$PCT" "MEMRA_METRICS_TOKEN=$METRICS_TOKEN" "$BIN" >"$1" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            echo "server died during boot:"
            tail -20 "$1"
            return 1
        }
        sleep 2
    done
    echo "server never became ready"
    return 1
}
stop() {
    # Only the server this script spawned; it stays in the collector's process group.
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

# Six DISJOINT long prompts (each > PREFIX_CACHE_MIN_TOKENS=64 for every tokenizer here) so no
# two share a 64-token prefix, plus one extension for the strict-prefix hit shape.
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
P_C="Write the maintenance log template for a fleet of eight municipal snowplows. For each \
vehicle, in fleet-number order, record the plow blade type, the hydraulic fluid grade, the \
spreader calibration date, the tire tread depth in millimeters, the operator certification \
class, and the fault that would pull it from the next storm roster. Be systematic and terse; \
do not skip a vehicle. Close with a short paragraph on pre-season salt-brine tank checks."
P_D="Prepare the audit worksheet for a regional seed bank's cold rooms. For each of the six \
rooms, in numbering order, list the target temperature, the relative humidity band, the \
door-seal inspection date, the backup compressor status, the accession count stored, and \
the excursion that would trigger an emergency transfer. Be systematic and terse throughout. \
After the six rooms, add a short paragraph on desiccant replacement scheduling."
P_E="Compose the inspection sheet for a small harbor's mooring field. For each of the ten \
mooring blocks, in chart order, note the block mass in kilograms, the chain grade, the swivel \
condition, the pennant length, the last dive inspection date, and the wear that would condemn \
the assembly before the next season. Be systematic and terse; do not skip a block. Finish \
with a short paragraph on storm-season pre-checks of the whole field."
P_F="Set out the calibration register for a university's seismograph array. For each of the nine \
stations, in array order, give the sensor model class, the sampling rate in hertz, the digitizer \
gain setting, the last timing-reference check, the responsible operator role, and the drift \
that would take the station out of the network solution. Be systematic and terse; do not skip a \
station. End with a short paragraph on the shared timing reference and its failure modes."
EXT=" Finally, state which single item most often fails first and why."

req() { # $1 bearer-key $2 prompt $3 out-json
    python3 - "$PORT" "$1" "$2" "$3" <<'PY'
import json, sys, urllib.request, urllib.error
port, key, prompt, out = sys.argv[1:5]
body = {"model": "gate", "prompt": prompt, "max_tokens": 48, "temperature": 0, "cache_salt": "s1"}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
)
try:
    r = urllib.request.urlopen(req, timeout=600)
    doc = json.load(r)
    doc["_http_status"] = r.status
except urllib.error.HTTPError as e:
    doc = {"_http_status": e.code, "_body": e.read().decode(errors="replace")}
json.dump(doc, open(out, "w"), indent=1)
PY
}

scrape_metrics() { # $1 out-json
    python3 - "$PORT" "$METRICS_TOKEN" "$1" <<'PY'
import json, sys, urllib.request, urllib.error
port, token, out = sys.argv[1:4]
req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {token}"})
try:
    r = urllib.request.urlopen(req, timeout=30)
    doc = json.load(r)
    doc["_http_status"] = r.status
except urllib.error.HTTPError as e:
    doc = {"_http_status": e.code, "_body": e.read().decode(errors="replace")}
json.dump(doc, open(out, "w"), indent=1)
PY
}

FAILS=0
# Matcher self-check (the attempt-2 lesson): the tag must match as a literal, never as a set.
printf '%s\n' "[prefix-host] demote: 1 tokens, 1.0MB" > "$EV/.matcher-probe"
grep -qE -- "\[prefix-host\] demote: " "$EV/.matcher-probe" || { echo "REFUSED: ERE matcher does not match the literal tag" >&2; exit 2; }
rm -f "$EV/.matcher-probe"
chk() {
    local name=$1
    shift
    if "$@"; then echo "  ok: $name"; else
        echo "  FAIL: $name"
        FAILS=$((FAILS + 1))
    fi
}
# ERE on purpose: in basic regex `[prefix-host]` is a bracket expression and never matches the
# literal tag (attempt 2 of the day-12 cell: two `present` FAILs and two vacuous `absent` passes).
absent() { ! grep -qE -- "$1" "$2"; }
present() { grep -qE -- "$1" "$2"; }
jqpy() { # $1 file $2 python-expr over loaded json `r`
    python3 -c "
import json, sys
r = json.load(open('$1'))
sys.exit(0 if ($2) else 1)"
}

echo "== $ARM arm boot (host $HOST_MB MiB, tenant share $PCT%, device $CACHE_MB MiB, two keyring tenants) =="
sha256sum "$BIN" | tee "$EV/binary.sha256"
boot "$EV/server.log"
req "$KEY_BETA" "$P_D" "$EV/r1.json"
req "$KEY_ACME" "$P_A" "$EV/r2.json"
req "$KEY_ACME" "$P_B" "$EV/r3.json"
req "$KEY_ACME" "$P_C" "$EV/r4.json"
req "$KEY_ACME" "$P_E" "$EV/r5.json"
req "$KEY_ACME" "$P_F" "$EV/r6.json"
req "$KEY_BETA" "$P_D$EXT" "$EV/r7.json"
req "$KEY_ACME" "$P_F$EXT" "$EV/r8.json"
scrape_metrics "$EV/metrics.json"
stop
LOG=$EV/server.log

echo "== both arms: shape =="
for i in 1 2 3 4 5 6 7 8; do
    chk "r$i HTTP 200" jqpy "$EV/r$i.json" "r.get('_http_status') == 200"
done
chk "/metrics scraped with the operator token (200, process-wide counters present)" jqpy "$EV/metrics.json" "r.get('_http_status') == 200 and 'prefix_host_tenant_rejects' in r"
chk "no device-tier promote-insert skip (the device budget holds a promote beside a protected entry)" absent "skip pinned host-promote insert" "$LOG"
chk "the tier is on with the configured share cap" present "tenant share cap $PCT% = " "$LOG"
chk "the pool never filled: no LRU eviction" absent "\[prefix-host\] evict \(LRU\)" "$LOG"
chk "no allocation refusal" absent "admission refused" "$LOG"
chk "no host-tier refusal or failure line" absent "\[prefix-host\] REFUSED|\[prefix-host\] demote failed|\[prefix-host\] demote refused|TIER DISABLED" "$LOG"
chk "beta's first demote landed" present "\[prefix-host\] demote: .*t:beta" "$LOG"
chk "acme's first two demotes landed (r3, r4)" python3 -c "
import re, sys
n = sum(1 for l in open('$LOG') if l.startswith('[prefix-host] demote: ') and 't:acme' in l)
sys.exit(0 if n >= 2 else 1)"
chk "beta's entry promotes at r7 in this arm: acme's cap did not touch beta's row" python3 -c "
import json, sys
r1 = json.load(open('$EV/r1.json'))['usage']['prompt_tokens']
r7 = json.load(open('$EV/r7.json'))['usage']['prompt_tokens_details']['cached_tokens']
sys.exit(0 if r7 == r1 and r1 > 0 else 1)"
chk "the r7 promote line names beta" present "\[prefix-host\] promote: .*t:beta" "$LOG"
chk "no tenant-share eviction ever names beta" absent "evict \(tenant share\): .*t:beta" "$LOG"
chk "every tenant-share eviction names acme, in both the row and the namespace" python3 -c "
import sys
bad = [l for l in open('$LOG') if 'evict (tenant share): ' in l and not (\"tenant \\\"t:acme\\\"\" in l and 'ns \"t:acme' in l)]
sys.exit(1 if bad else 0)"

echo "== $ARM arm: the verdict lines =="
EVAP='demote evaporated at the tenant share cap before the D2H copy'
RECLAIM='\[prefix-host\] evict \(tenant share\): '
if [ "$ARM" = base ]; then
    chk "base: acme's third demotion EVAPORATED at the cap (the pre-#384 line)" present "$EVAP: .*t:acme" "$LOG"
    chk "base: nothing of acme's own was evicted" absent "$RECLAIM" "$LOG"
    chk "base: r8 is cold (the evaporated entry is gone)" jqpy "$EV/r8.json" "r['usage']['prompt_tokens_details']['cached_tokens'] == 0"
    chk "base: /metrics counts the rejects" jqpy "$EV/metrics.json" "r.get('prefix_host_tenant_rejects', 0) >= 1"
    chk "base: /metrics has no reclaims" jqpy "$EV/metrics.json" "r.get('prefix_host_tenant_reclaims', 0) == 0"
else
    chk "fix: no evaporation line" absent "$EVAP" "$LOG"
    chk "fix: acme's own oldest entry was evicted at the cap" present "$RECLAIM.*t:acme" "$LOG"
    chk "fix: every tenant-share eviction is followed by acme's own demote landing (D2H after the reclaim)" python3 -c "
import sys
lines = [l.rstrip('\n') for l in open('$LOG')]
ok = True
for i, l in enumerate(lines):
    if 'evict (tenant share): ' not in l:
        continue
    tail = lines[i + 1:i + 8]
    landed = [t for t in tail if t.startswith('[prefix-host] demote: ') and 't:acme' in t]
    ok &= bool(landed)
sys.exit(0 if ok else 1)"
    chk "fix: r8 promotes the reclaimed admission (cached_tokens == r6 prompt_tokens)" python3 -c "
import json, sys
r6 = json.load(open('$EV/r6.json'))['usage']['prompt_tokens']
r8 = json.load(open('$EV/r8.json'))['usage']['prompt_tokens_details']['cached_tokens']
sys.exit(0 if r8 == r6 and r6 > 0 else 1)"
    chk "fix: the r8 promote line names acme" present "\[prefix-host\] promote: .*t:acme" "$LOG"
    chk "fix: /metrics counts the reclaims and no rejects" jqpy "$EV/metrics.json" "r.get('prefix_host_tenant_reclaims', 0) >= 1 and r.get('prefix_host_tenant_rejects', 0) == 0"
fi

# The replay verifier's input: the facts of this arm, one JSON.
python3 - "$ARM" "$EV" "$LOG" <<'PY'
import hashlib, json, re, sys
arm, ev, log = sys.argv[1:4]
lines = [l.rstrip("\n") for l in open(log, errors="replace")]
def pick(pat):
    return [l for l in lines if re.search(pat, l)]
def strip_ms(l):
    return re.sub(r" in [\d.]+ms", "", l)
reqs = {}
for i in range(1, 9):
    d = json.load(open(f"{ev}/r{i}.json"))
    u = d.get("usage", {})
    text = d["choices"][0]["text"] if d.get("choices") else None
    reqs[f"r{i}"] = {
        "http": d.get("_http_status"),
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
    }
m = json.load(open(f"{ev}/metrics.json"))
json.dump({
    "arm": arm,
    "binary_sha256": open(f"{ev}/binary.sha256").read().split()[0],
    "requests": reqs,
    "demotes": [strip_ms(l) for l in pick(r"^\[prefix-host\] demote: ")],
    "evaporations": pick(r"demote evaporated at the tenant share cap before the D2H copy"),
    "reclaims": pick(r"^\[prefix-host\] evict \(tenant share\): "),
    "promotes": [strip_ms(l) for l in pick(r"^\[prefix-host\] promote: ")],
    "lru_evictions": pick(r"^\[prefix-host\] evict \(LRU\)"),
    "device_promote_skips": pick(r"skip pinned host-promote insert"),
    "metrics": {k: m.get(k) for k in ("prefix_host_entries", "prefix_host_bytes", "prefix_host_demotions",
                                       "prefix_host_promotions", "prefix_host_tenant_rejects",
                                       "prefix_host_tenant_reclaims", "prefix_host_rejected_allocs")},
}, open(f"{ev}/SUMMARY.json", "w"), indent=1)
PY

if [ "$FAILS" = 0 ]; then
    echo "GATE: kv-host-tenant-reclaim ($ARM arm) PASS"
    exit 0
fi
echo "GATE: kv-host-tenant-reclaim ($ARM arm) FAIL ($FAILS assertions)"
exit 1
