#!/usr/bin/env bash
# kv-host-spill-identity-gate.sh: the prefix-cache HOST TIER must round-trip entries
# byte-losslessly (lane/kv-host-spill-20260830). This is the cached-vs-fresh identity gate
# with a host-tier arm: a request served through a DEMOTE -> PROMOTE -> restore round trip
# must be byte-identical to the same request served cold on a tier-off boot of the SAME
# binary. Greedy is the instrument here (byte-determinism), never the product; the pod
# battery adds the vendor-default sampled probe separately per the serving law.
#
# Cell design (one PC-ISO namespace, budget forced so the round trip HAS to happen):
#   ON boot: MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_VERIFY=1, device budget
#     MEMRA_PREFIX_CACHE_MB=$CACHE_MB sized to hold ONE seed entry but not two.
#     r1  P_A cold        -> seeds entry E_A
#     r2  P_B cold        -> seeds E_B; the byte budget evicts E_A, which must DEMOTE
#                            ([prefix-host] demote line, digest recorded)
#     r3  P_A + EXT       -> continuation pool misses (different suffix), device pool
#                            misses (E_A is gone), host PROMOTES E_A ([prefix-host]
#                            promote + verify ok lines) and the suffix restores:
#                            0 < cached_tokens < prompt_tokens
#     r4  P_A + EXT again -> deterministic repeat on the promoted path: text == r3
#   OFF boot (MEMRA_KV_HOST_MB=0, same device budget): r1/r2/r3 again; r3 re-primes cold
#     (cached_tokens == 0).
#   IDENTITY LAW: text(on r3) == text(off r3) and text(on r1) == text(off r1): the host
#   round trip must not change a single byte.
#
# MEMRA_HOSTGATE_TEETH=1 is the FORCED-TINY RED ARM (the verdict must invert): the ON boot
# takes MEMRA_KV_HOST_MB=1 (a 1 MiB tier no real entry fits), and the gate then REQUIRES
# the opposite behavior: a named "skip demote" refusal, ZERO promotions, r3 cold
# (cached_tokens == 0) yet still byte-identical to the OFF boot. A binary whose host tier
# does nothing passes teeth and FAILS the default arm, which is what gives the default arm
# its teeth.
#
# usage: kv-host-spill-identity-gate.sh <model.gguf> <server_bin> <evidence_dir>
# env:   MEMRA_HOSTGATE_CACHE_MB (default 1024)  device prefix budget; must hold ONE seed
#                                                entry but not two: the gate FAILS with a
#                                                tuning message if no demote fires
#        MEMRA_HOSTGATE_HOST_MB  (default 8192)  host tier budget for the ON boot
#        MEMRA_HOSTGATE_TEETH=1                  forced-tiny red arm (see above)
# Boots its own servers one arm at a time (flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}).
# Exit 0 = every assertion held. Evidence: <evidence_dir>/host-{on,off}-r{1..4}.json + logs.
set -euo pipefail
MODEL=$1
BIN=$2
EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18109}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "kv-host-spill-identity-gate: FAIL: $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
mkdir -p "$EV"
SERVER_PID=""
CACHE_MB=${MEMRA_HOSTGATE_CACHE_MB:-1024}
HOST_MB=${MEMRA_HOSTGATE_HOST_MB:-8192}
TEETH=${MEMRA_HOSTGATE_TEETH:-0}

boot() { # $1 extra-env-string  $2 log
    memra_port_guard kv-host-spill-identity-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving: refusing to boot over it"
        return 1
    fi
    # shellcheck disable=SC2086
    flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" $1 "$BIN" >"$2" 2>&1 &
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
    pkill -x memra-server 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
            SERVER_PID=""
            sleep 2
            return 0
        }
        sleep 1
    done
    pkill -9 -x memra-server 2>/dev/null || true
    SERVER_PID=""
    sleep 3
}
trap stop EXIT

# Two DISJOINT long prompts (both > PREFIX_CACHE_MIN_TOKENS=64 for every tokenizer here) so
# E_B shares no 64-token prefix with E_A, plus an extension for the strict-prefix hit shape.
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
EXT=" Finally, state which single item most often fails first and why."

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

scrape_metrics() { # $1 out-json
    python3 - "$PORT" "$1" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
r = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30)
json.dump(json.load(r), open(out, "w"), indent=1)
PY
}

FAILS=0
# chk NAME CMD...: run CMD under an `if` so set -e never fires on an asserted failure;
# a failing assertion is COUNTED, never silently fatal (loud-failures law).
chk() {
    local name=$1
    shift
    if "$@"; then echo "  ok: $name"; else
        echo "  FAIL: $name"
        FAILS=$((FAILS + 1))
    fi
}
absent() { ! grep -q "$1" "$2"; }
jqpy() { # $1 file $2 python-expr over loaded json `r`
    python3 -c "
import json, sys
r = json.load(open('$1'))
sys.exit(0 if ($2) else 1)"
}
text_eq() { # $1 file $2 file
    python3 -c "
import json, sys
a = json.load(open('$1'))['choices'][0]['text']
b = json.load(open('$2'))['choices'][0]['text']
sys.exit(0 if a == b else 1)"
}

if [ "$TEETH" = 1 ]; then
    echo "== host-tier ON boot, FORCED-TINY RED ARM (MEMRA_KV_HOST_MB=1) =="
    boot "MEMRA_KV_HOST_MB=1 MEMRA_KV_HOST_VERIFY=1" "$EV/host-on-server.log"
else
    echo "== host-tier ON boot (MEMRA_KV_HOST_MB=$HOST_MB, verify on) =="
    boot "MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_VERIFY=1" "$EV/host-on-server.log"
fi
req "$P_A" "$EV/host-on-r1.json"
req "$P_B" "$EV/host-on-r2.json"
req "$P_A$EXT" "$EV/host-on-r3.json"
req "$P_A$EXT" "$EV/host-on-r4.json"
scrape_metrics "$EV/host-on-metrics.json"
stop

# The device budget must have forced the eviction that feeds the tier: in EITHER arm.
# The good path is a plain echo, NOT `chk` (this branch keeps its multi-line tuning hint
# on failure, which chk's uniform FAIL line would drop). It was `ok "..." 0`, an
# UNDEFINED helper, so under `set -e` the gate ABORTED on its GOOD path, right after the
# ON arm and before a single assertion ran (crash-on-success, battery-20260831 Q1;
# verbatim crash + on-box 1-line patch banked in darklanes
# research/kv-fastband-20260830/battery-20260831/raw/q38-identity/).
if grep -q "\[prefix-cache\] evict" "$EV/host-on-server.log"; then
    echo "  ok: device budget forced an eviction (the tier's feed)"
else
    echo "  FAIL: no device eviction fired: MEMRA_HOSTGATE_CACHE_MB=$CACHE_MB holds both"
    echo "        seed entries (or refused one outright); tune it to hold exactly one"
    FAILS=$((FAILS + 1))
fi

if [ "$TEETH" = 1 ]; then
    # RED ARM: the tiny tier must refuse BY NAME, promote nothing, and change no bytes.
    chk "teeth: demote refused by name (entry > 1 MiB host budget)" \
        grep -q "\[prefix-host\] skip demote: entry" "$EV/host-on-server.log"
    chk "teeth: no promote happened" \
        absent "\[prefix-host\] promote:" "$EV/host-on-server.log"
    chk "teeth: metrics agree (0 demotions, 0 promotions)" jqpy "$EV/host-on-metrics.json" \
        "r['prefix_host_demotions'] == 0 and r['prefix_host_promotions'] == 0"
    chk "teeth: r3 re-primed cold (no tier to hit)" jqpy "$EV/host-on-r3.json" \
        "r['usage']['prompt_tokens_details']['cached_tokens'] == 0"
else
    chk "E_A demoted to the host tier (named log line)" \
        grep -q "\[prefix-host\] demote:" "$EV/host-on-server.log"
    chk "E_A promoted back on the r3 probe (named log line)" \
        grep -q "\[prefix-host\] promote:" "$EV/host-on-server.log"
    chk "MEMRA_KV_HOST_VERIFY digest matched across the round trip" \
        grep -q "\[prefix-host\] verify ok" "$EV/host-on-server.log"
    chk "no verify mismatch anywhere in the run" \
        absent "\[prefix-host\] VERIFY FAILED" "$EV/host-on-server.log"
    chk "metrics: demotions >= 1, promotions >= 1, zero rejected allocs" \
        jqpy "$EV/host-on-metrics.json" \
        "r['prefix_host_demotions'] >= 1 and r['prefix_host_promotions'] >= 1 and r['prefix_host_rejected_allocs'] == 0"
    chk "r3 served a strict-prefix hit through the promoted entry" \
        jqpy "$EV/host-on-r3.json" \
        "0 < r['usage']['prompt_tokens_details']['cached_tokens'] < r['usage']['prompt_tokens']"
    chk "r4 repeats r3 byte-for-byte (deterministic promoted path)" \
        text_eq "$EV/host-on-r3.json" "$EV/host-on-r4.json"
fi

echo "== host-tier OFF twin boot (MEMRA_KV_HOST_MB=0: the rollback seam) =="
boot "MEMRA_KV_HOST_MB=0" "$EV/host-off-server.log"
req "$P_A" "$EV/host-off-r1.json"
req "$P_B" "$EV/host-off-r2.json"
req "$P_A$EXT" "$EV/host-off-r3.json"
stop

chk "OFF boot never touches the tier (no [prefix-host] line at all)" \
    absent "\[prefix-host\]" "$EV/host-off-server.log"
chk "OFF r3 re-primed cold (nothing preserved the evicted entry)" \
    jqpy "$EV/host-off-r3.json" \
    "r['usage']['prompt_tokens_details']['cached_tokens'] == 0"
# THE IDENTITY LAW: the host round trip must not change a single byte.
chk "r1 ON == OFF byte identity (cold path unperturbed)" \
    text_eq "$EV/host-on-r1.json" "$EV/host-off-r1.json"
chk "r3 ON == OFF byte identity (promoted restore == cold re-prime)" \
    text_eq "$EV/host-on-r3.json" "$EV/host-off-r3.json"

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=$TEETH)"
else
    echo "KV-HOST-SPILL IDENTITY GATE: $FAILS FAILURE(S) (teeth=$TEETH)"
    exit 1
fi
