#!/usr/bin/env bash
# Same-binary fresh-boot default/rollback golden and route-trace diagnostic.
set -euo pipefail

REPO=${SIGROUTER_REPO:-/home/ubuntu/memra-cx-sigrouter}
OUT=${SIGROUTER_GOLDEN_AB_OUT:-$REPO/research/sigrouter-20260811/raw/box1-golden-ab}
MODEL_ROOT=${SIGROUTER_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf
GOLDEN=/home/ubuntu/darktrain2/golden-response.bin
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PORT=${SIGROUTER_GOLDEN_AB_PORT:-18455}
BASE=http://127.0.0.1:$PORT

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

wait_idle() {
    for _ in $(seq 1 120); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    tail -200 "$log"
    return 1
}

run_arm() {
    local arm=$1
    local cell="$OUT/$arm" log="$OUT/$arm/server.log"
    local -a policy=()
    [[ $arm == rollback ]] && policy=(MEMRA_SIG_ROUTER=0)
    mkdir -p "$cell"
    echo "arm=$arm start=$(date -u +%FT%TZ)"
    env \
        -u MEMRA_SIG_ROUTER \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_K \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_BG_JOB \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="step37=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PRIME_BATCH_HOLD_MS=4 \
        MEMRA_MOE_TRACE="$cell/moe-ids.log" \
        MEMRA_MOE_WEIGHT_TRACE="$cell/moe-weights.log" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    set +e
    "$QOS" \
        --base "$BASE" \
        --model step37 \
        --label "sigrouter-$arm" \
        --requests 1 \
        --max-tokens 64 \
        --golden "$GOLDEN" \
        --rows "$cell/qos-rows.jsonl" \
        --summary "$cell/qos-summary.json" \
        2>&1 | tee "$cell/qos.log"
    local probe_rc=${PIPESTATUS[0]}
    set -e
    echo "$probe_rc" >"$cell/qos.rc"
    case "$probe_rc" in 0|86) ;; *) return "$probe_rc" ;; esac
    stop_server
    if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal|illegal' "$log"; then
        echo "FAIL: server failure signature in $log"
        return 1
    fi
    echo "arm=$arm probe_rc=$probe_rc done=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GOLDEN_AB_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" >"$OUT/SHA256SUMS"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_arm default
run_arm rollback

if cmp -s "$OUT/default/moe-ids.log" "$OUT/rollback/moe-ids.log"; then
    echo "route_ids=IDENTICAL"
else
    echo "route_ids=DIFFERENT"
    diff -u "$OUT/rollback/moe-ids.log" "$OUT/default/moe-ids.log" \
        >"$OUT/moe-ids.diff" || true
fi
if cmp -s "$OUT/default/moe-weights.log" "$OUT/rollback/moe-weights.log"; then
    echo "route_weights_9dp=IDENTICAL"
else
    echo "route_weights_9dp=DIFFERENT"
    diff -u "$OUT/rollback/moe-weights.log" "$OUT/default/moe-weights.log" \
        >"$OUT/moe-weights.diff" || true
fi
echo "GOLDEN_AB_DONE $(date -u +%FT%TZ)"
