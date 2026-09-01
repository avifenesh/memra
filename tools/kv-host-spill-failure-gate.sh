#!/usr/bin/env bash
# kv-host-spill-failure-gate.sh: the prefix-cache HOST TIER's failure paths must be LOUD and
# harmless (lane/kv-host-spill-20260830). Every failure path here is EXECUTED, not asserted
# in prose (the loud-failures-fail-quietly law), via the MEMRA_KV_HOST_FAULT diagnostic door
# (docs/FLAGS.md section 4). Three cells, each its own boot:
#
#   pool-full      MEMRA_KV_HOST_MB=1: a real entry cannot fit the 1 MiB tier, so the demote
#                  must refuse BY NAME ("skip demote: entry ... > host budget"), keep zero
#                  host entries, and leave serving untouched.
#   digest-mismatch MEMRA_KV_HOST_VERIFY=1 MEMRA_KV_HOST_FAULT=flip-demote: one demoted K
#                  byte is flipped AFTER the demote digest is recorded, so the promote must
#                  print "[prefix-host] VERIFY FAILED", drop the host entry, and serve the
#                  request cold with the SAME bytes as the reference cell.
#   alloc-refusal  MEMRA_KV_HOST_FAULT=alloc-fail: every pinned alloc reports failure, so the
#                  first demote must print "[prefix-host] TIER DISABLED" (latched off, no
#                  pageable fallback), count prefix_host_rejected_allocs, complete zero
#                  demotions, and leave serving untouched.
#
# HARMLESSNESS ORACLE: cell 1 (pool-full) doubles as the byte reference; its r3 is a cold
# re-prime of P_A+EXT, and cells 2 and 3 must reproduce those bytes exactly (greedy is the
# instrument; same binary, same model, cross-boot determinism as in the other cache gates).
#
# usage: kv-host-spill-failure-gate.sh <model.gguf> <server_bin> <evidence_dir>
# env:   MEMRA_HOSTGATE_CACHE_MB (default 1024)  device prefix budget; must hold ONE seed
#                                                entry but not two, or no demote ever fires
#        MEMRA_HOSTGATE_HOST_MB  (default 8192)  host budget for the fault cells
# Boots its own servers one cell at a time (flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}).
# Exit 0 = every assertion held. Evidence: <evidence_dir>/<cell>-r{1..3}.json + logs.
set -euo pipefail
MODEL=$1
BIN=$2
EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18119}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "kv-host-spill-failure-gate: FAIL, $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
mkdir -p "$EV"
SERVER_PID=""
CACHE_MB=${MEMRA_HOSTGATE_CACHE_MB:-1024}
HOST_MB=${MEMRA_HOSTGATE_HOST_MB:-8192}

boot() { # $1 extra-env-string  $2 log
    memra_port_guard kv-host-spill-failure-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"
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

# Same fixtures as kv-host-spill-identity-gate.sh: two disjoint long prompts + an extension.
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
chk() { # NAME CMD...: run CMD under `if` so set -e never fires on an asserted failure
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
run_cell() { # $1 cell-name: r1 seeds, r2 forces the eviction/demote, r3 probes P_A+EXT
    req "$P_A" "$EV/$1-r1.json"
    req "$P_B" "$EV/$1-r2.json"
    req "$P_A$EXT" "$EV/$1-r3.json"
    scrape_metrics "$EV/$1-metrics.json"
}

echo "== cell 1: POOL-FULL (MEMRA_KV_HOST_MB=1, no real entry fits) =="
boot "MEMRA_KV_HOST_MB=1" "$EV/poolfull-server.log"
run_cell poolfull
stop
chk "device budget forced an eviction (the failure path's trigger)" \
    grep -q "\[prefix-cache\] evict" "$EV/poolfull-server.log"
chk "pool-full refusal is LOUD and named" \
    grep -q "\[prefix-host\] skip demote: entry" "$EV/poolfull-server.log"
chk "nothing entered the tier" \
    jqpy "$EV/poolfull-metrics.json" \
    "r['prefix_host_entries'] == 0 and r['prefix_host_demotions'] == 0 and r['prefix_host_promotions'] == 0"
chk "r3 still serves (cold re-prime, zero cached)" \
    jqpy "$EV/poolfull-r3.json" \
    "r['usage']['prompt_tokens_details']['cached_tokens'] == 0 and len(r['choices'][0]['text']) > 0"

echo "== cell 2: DIGEST MISMATCH (verify on + MEMRA_KV_HOST_FAULT=flip-demote) =="
boot "MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_VERIFY=1 MEMRA_KV_HOST_FAULT=flip-demote" \
    "$EV/digest-server.log"
run_cell digest
stop
chk "the fault door announced the injected corruption" \
    grep -q "\[prefix-host\] FAULT: flipped one demoted K byte" "$EV/digest-server.log"
chk "the corrupted entry DEMOTED (the mismatch needs a resident entry)" \
    grep -q "\[prefix-host\] demote:" "$EV/digest-server.log"
chk "the promote caught it: VERIFY FAILED, loud and named" \
    grep -q "\[prefix-host\] VERIFY FAILED" "$EV/digest-server.log"
chk "no successful promote happened" \
    absent "\[prefix-host\] promote:" "$EV/digest-server.log"
chk "metrics: zero promotions after the refusal" \
    jqpy "$EV/digest-metrics.json" "r['prefix_host_promotions'] == 0"
chk "r3 served the COLD path with reference bytes (corruption never reached a customer)" \
    text_eq "$EV/digest-r3.json" "$EV/poolfull-r3.json"

echo "== cell 3: PINNED-ALLOC REFUSAL (MEMRA_KV_HOST_FAULT=alloc-fail) =="
boot "MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_FAULT=alloc-fail" "$EV/alloc-server.log"
run_cell alloc
stop
chk "the tier LATCHED OFF loudly on the first alloc failure" \
    grep -q "\[prefix-host\] TIER DISABLED" "$EV/alloc-server.log"
chk "the refusal counted" \
    jqpy "$EV/alloc-metrics.json" "r['prefix_host_rejected_allocs'] >= 1"
chk "no silent pageable fallback: zero demotions completed" \
    jqpy "$EV/alloc-metrics.json" \
    "r['prefix_host_demotions'] == 0 and r['prefix_host_entries'] == 0"
chk "r3 served the cold path with reference bytes" \
    text_eq "$EV/alloc-r3.json" "$EV/poolfull-r3.json"

if [ "$FAILS" -eq 0 ]; then
    echo "KV-HOST-SPILL FAILURE GATE: ALL GREEN"
else
    echo "KV-HOST-SPILL FAILURE GATE: $FAILS FAILURE(S)"
    exit 1
fi
