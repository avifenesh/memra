#!/usr/bin/env bash
# Production-gateway preflight and scored two-hour Q27 soak on the eu-west PRO pair.
set -Eeuo pipefail
umask 027

MODE=${1:-}
case "$MODE" in
    preflight|soak) ;;
    *) echo "usage: $0 preflight|soak" >&2; exit 2 ;;
esac

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${GATEWAY_ROOT:-/opt/scratch/nvme/cx-gateway}
REPO=${GATEWAY_REPO:-$ROOT/memra}
MODEL_ROOT=${GATEWAY_MODEL_ROOT:-$ROOT/models}
MODEL=${GATEWAY_MODEL:-$MODEL_ROOT/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
MODEL_SOURCE=${GATEWAY_MODEL_SOURCE:-$REPO/deploy/gateway/q27-artifact.manifest}
PYTHON=${GATEWAY_PROBE_PYTHON:-$ROOT/probe-venv/bin/python}
SERVER=$REPO/target/release/memra-server
PROBE=$REPO/deploy/gateway/probe.py
METADATA=$REPO/deploy/gateway/q27-models.toml
CAPTURE=$REPO/deploy/gateway/capture-manifest.sh
EXPECTED_COMMIT=${GATEWAY_EXPECTED_COMMIT:-}
EXPECTED_MODEL=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
MODEL_ID=qwen/qwen3.6-27b
PORT0=${GATEWAY_PORT0:-${GATEWAY_PORT:-18427}}
PORT1=${GATEWAY_PORT1:-18428}
PORTS=("$PORT0" "$PORT1")
BASES=("http://127.0.0.1:$PORT0" "http://127.0.0.1:$PORT1")
DURATION=0
[[ "$MODE" == soak ]] && DURATION=7200
STAMP=${GATEWAY_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${GATEWAY_OUT:-$ROOT/raw/$MODE-$STAMP}

[[ -n "$EXPECTED_COMMIT" ]] || {
    echo "GATEWAY_EXPECTED_COMMIT is required" >&2
    exit 2
}
[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 1; }
install -d -m 0750 "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pids=()
gpu_sampler_pid=
vmstat_pid=
secret_dir=

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "timestamp=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

stop_servers() {
    local pid _ alive rc=0
    ((${#server_pids[@]})) || return 0
    for pid in "${server_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]}"; do
        alive=1
        for _ in $(seq 1 180); do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                alive=0
                break
            fi
            sleep 1
        done
        if ((alive)); then
            echo "FAIL: owned server $pid did not drain in 180 seconds"
            kill -KILL "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            rc=1
        fi
    done
    server_pids=()
    return "$rc"
}

stop_samplers() {
    local pid
    for pid in "${gpu_sampler_pid:-}" "${vmstat_pid:-}"; do
        [[ -n "$pid" ]] || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    gpu_sampler_pid=
    vmstat_pid=
}

cleanup() {
    local rc=$?
    stop_servers || rc=1
    stop_samplers
    if [[ -n "${secret_dir:-}" && -d "$secret_dir" ]]; then
        rm -r -- "$secret_dir"
    fi
    snapshot "$OUT/gpu-cleanup.log" cleanup || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

write_manifest() {
    local temp
    temp=$(mktemp "$OUT/.manifest.XXXXXX")
    (
        cd "$OUT"
        find . -type f ! -name MANIFEST.sha256 ! -name driver.log ! -name '.manifest.*' \
            -print0 | sort -z | xargs -0 sha256sum
    ) >"$temp"
    mv "$temp" "$OUT/MANIFEST.sha256"
}

echo "GATEWAY_RUN_START mode=$MODE timestamp=$(date -u +%FT%TZ) out=$OUT"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
    git -C "$REPO" status --short
    echo "FAIL: staged checkout is dirty"
    exit 1
}
[[ -x "$SERVER" && -x "$PROBE" && -x "$CAPTURE" && -x "$PYTHON" ]]
"$PYTHON" -c 'import jsonschema; print(jsonschema.__version__)' \
    >"$OUT/jsonschema-version.txt" 2>&1
"$PYTHON" - "$PROBE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
compile(path.read_text(), str(path), "exec")
PY
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$EXPECTED_MODEL" ]]

for port in "${PORTS[@]}"; do
    if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
        echo "FAIL: port $port already has a listener"
        ss -tlnp 2>/dev/null | grep "[:.]$port " || true
        exit 1
    fi
done

{
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "mode=$MODE"
    echo "duration_s=$DURATION"
    echo "runtime_commit=$EXPECTED_COMMIT"
    echo "model_id=$MODEL_ID"
    echo "model_path=$MODEL"
    echo "model_sha256=$EXPECTED_MODEL"
    echo "model_bytes=$(stat -c %s "$MODEL")"
    echo "model_source_manifest=$MODEL_SOURCE"
    echo "shape=two independent Q27 replicas on physical GPU0 and GPU1; c=4 tenant cap per replica; spec off; ctx8192; prefix cache 4096 MiB per replica"
    echo "traffic=the same streaming plus non-streaming, tools, strict structured output, exact 90%-hit, and overload soak runs concurrently against both replicas"
    hostname
    uname -a
    git -C "$REPO" log -5 --oneline --decorate
    sha256sum "$SERVER" "$PROBE" "$METADATA" "$0"
    [[ -f "$MODEL_SOURCE" ]] && sha256sum "$MODEL_SOURCE"
} >"$OUT/provenance.txt"
cp "$MODEL_SOURCE" "$OUT/model-source.txt" 2>/dev/null || true
git -C "$REPO" status --porcelain=v1 --untracked-files=all >"$OUT/git-status.txt"
env | cut -d= -f1 | LC_ALL=C sort >"$OUT/environment-variable-names.txt"
snapshot "$OUT/gpu-queue-check.log" queue-check
fuser -v /tmp/memra-gpu.lock >"$OUT/flock-before.txt" 2>&1 || true

exec 9>/tmp/memra-gpu.lock
flock -w 21600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GATEWAY_LOCK_ACQUIRED timestamp=$(date -u +%FT%TZ) pid=$$"
snapshot "$OUT/gpu-before.log" before
apps=$(compute_apps)
[[ -z "$apps" ]] || { echo "$apps"; echo "FAIL: GPU pair is not idle"; exit 1; }

secret_dir=$(mktemp -d /tmp/cx-gateway-keys.XXXXXX)
chmod 0700 "$secret_dir"
keyring=$secret_dir/keys.toml
primary_key=$secret_dir/primary.key
isolation_key=$secret_dir/isolation.key
"$SERVER" --gen-key gateway_soak --rate-limit 4 --keys "$keyring" >"$primary_key"
"$SERVER" --gen-key gateway_isolation --rate-limit 4 --keys "$keyring" >"$isolation_key"
chmod 0600 "$keyring" "$primary_key" "$isolation_key"
{
    echo "keyring_sha256=$(sha256sum "$keyring" | awk '{print $1}')"
    echo "keyring_mode=$(stat -c %a "$keyring")"
    echo "tenants=gateway_soak,gateway_isolation"
    echo "tenant_cap=4"
    echo "plaintext_keys_retained=false"
} >"$OUT/keyring-manifest.txt"

nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -l 5 >"$OUT/gpu-5s.csv" 2>&1 &
gpu_sampler_pid=$!
vmstat 5 >"$OUT/vmstat-5s.log" 2>&1 &
vmstat_pid=$!

ledgers=("$OUT/request-cost-replica0.jsonl" "$OUT/request-cost-replica1.jsonl")
server_logs=("$OUT/server-replica0.log" "$OUT/server-replica1.log")
for replica in 0 1; do
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST -u MEMRA_MOE_RESIDENT \
        -u MEMRA_MOE_RESIDENT_GB -u MEMRA_METRICS_TOKEN \
        CUDA_VISIBLE_DEVICES="$replica" \
        MEMRA_MODELS="$MODEL_ID=$MODEL" \
        MEMRA_MODEL_METADATA="$METADATA" \
        MEMRA_REQUEST_LEDGER="${ledgers[$replica]}" \
        MEMRA_API_KEYS="$keyring" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:${PORTS[$replica]}" \
        MEMRA_TAG="cx-gateway-$MODE-r$replica" MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"${server_logs[$replica]}" 2>&1 &
    server_pids+=("$!")
done

ready=(0 0)
for _ in $(seq 1 900); do
    for replica in 0 1; do
        if [[ "${ready[$replica]}" == 0 ]] \
            && curl -sf "${BASES[$replica]}/readyz" \
                >"$OUT/ready-replica$replica.json" 2>/dev/null; then
            ready[replica]=1
        fi
        if ! kill -0 "${server_pids[$replica]}" 2>/dev/null; then
            echo "FAIL: replica $replica server died during boot"
            tail -200 "${server_logs[$replica]}"
            exit 1
        fi
    done
    [[ "${ready[0]}" == 1 && "${ready[1]}" == 1 ]] && break
    sleep 1
done
[[ "${ready[0]}" == 1 && "${ready[1]}" == 1 ]] || {
    echo "FAIL: both replicas did not become ready"
    exit 1
}
for replica in 0 1; do
    grep -q '\[prefix-cache\] on:' "${server_logs[$replica]}"
    grep -q '\[ledger\] durable request-cost ledger enabled:' "${server_logs[$replica]}"
done
ss -tlnp >"$OUT/listeners-after-start.txt" 2>&1
for replica in 0 1; do
    grep "[:.]${PORTS[$replica]} " "$OUT/listeners-after-start.txt" \
        | grep -q "pid=${server_pids[$replica]},"
done
snapshot "$OUT/gpu-ready.log" ready

run_probe() {
    local replica=$1 replica_out="$OUT/replica$1"
    install -d -m 0750 "$replica_out"
    timeout "$((DURATION + 14400))" "$PYTHON" "$PROBE" \
        --base-url "${BASES[$replica]}" \
        --model "$MODEL_ID" \
        --out "$replica_out/probe.jsonl" \
        --summary "$replica_out/summary.json" \
        --full \
        --duration "$DURATION" \
        --ledger "${ledgers[$replica]}" \
        --tenant gateway_soak \
        --isolation-tenant gateway_isolation \
        --api-key-file "$primary_key" \
        --isolation-api-key-file "$isolation_key" \
        --schema-out "$replica_out/openrouter-provider-schema-v2.4.json" \
        --models-out "$replica_out/models-openrouter.json" \
        --namespace "cx-gateway-$MODE-$STAMP-r$replica" \
        --timeout 1800 \
        >"$replica_out/probe.stdout" 2>"$replica_out/probe.stderr"
}

probe_pids=()
for replica in 0 1; do
    run_probe "$replica" &
    probe_pids+=("$!")
done
probe_rc=(0 0)
set +e
for replica in 0 1; do
    wait "${probe_pids[$replica]}"
    probe_rc[replica]=$?
done
set -e
echo "PROBES_DONE rc0=${probe_rc[0]} rc1=${probe_rc[1]} timestamp=$(date -u +%FT%TZ)"
[[ "${probe_rc[0]}" == 0 && "${probe_rc[1]}" == 0 ]]
for replica in 0 1; do
    jq -e '.verdict == "PASS" and .protocol_schema_billing_output_errors == 0' \
        "$OUT/replica$replica/summary.json" >/dev/null
    if [[ "$MODE" == soak ]]; then
        jq -e '.requested_duration_s == 7200 and .elapsed_s >= 7200' \
            "$OUT/replica$replica/summary.json" >/dev/null
    fi
done
jq -n \
    --arg mode "$MODE" \
    --slurpfile replica0 "$OUT/replica0/summary.json" \
    --slurpfile replica1 "$OUT/replica1/summary.json" \
    '{schema:"memra.gateway-pair-soak.v1", mode:$mode, verdict:"PASS",
      replicas:[$replica0[0],$replica1[0]],
      requested_duration_s:([$replica0[0].requested_duration_s,
                              $replica1[0].requested_duration_s] | min),
      elapsed_s:([$replica0[0].elapsed_s,$replica1[0].elapsed_s] | min),
      requests:($replica0[0].requests + $replica1[0].requests),
      protocol_schema_billing_output_errors:
        ($replica0[0].protocol_schema_billing_output_errors +
         $replica1[0].protocol_schema_billing_output_errors)}' \
    >"$OUT/summary.json"

stop_servers
stop_samplers
snapshot "$OUT/gpu-after.log" after
apps=$(compute_apps)
[[ -z "$apps" ]] || { echo "$apps"; echo "FAIL: GPU process remained after stop"; exit 1; }

for replica in 0 1; do
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]|\[ledger\] ERROR' \
        "${server_logs[$replica]}" \
        || grep -En 'MISMATCH' "${server_logs[$replica]}"; then
        echo "FAIL: replica $replica server log contains a fatal/error signature"
        exit 1
    fi
done

awk -F',' '
    { gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5) }
    $4 ~ /^[0-9.]+$/ && $5 ~ /^[0-9.]+$/ {
        n++
        temperature=$4 + 0
        power=$5 + 0
        if (n == 1 || temperature < tmin) tmin=temperature
        if (n == 1 || temperature > tmax) tmax=temperature
        if (n == 1 || power < pmin) pmin=power
        if (n == 1 || power > pmax) pmax=power
    }
    END {
        printf "samples=%d\ntemperature_min_c=%.0f\ntemperature_max_c=%.0f\npower_min_w=%.2f\npower_max_w=%.2f\n", n, tmin, tmax, pmin, pmax
    }
' "$OUT/gpu-5s.csv" >"$OUT/thermal-summary.txt"

for replica in 0 1; do
    MEMRA_REPO="$REPO" MEMRA_SERVER_BIN="$SERVER" \
    MEMRA_GATEWAY_MODEL="$MODEL" MEMRA_REQUEST_LEDGER="${ledgers[$replica]}" \
    MEMRA_MODEL_METADATA="$METADATA" MEMRA_API_KEYS="$keyring" \
        "$CAPTURE" --out "$OUT/offhost-manifest-replica$replica"
done

wc -l "$OUT/replica0/probe.jsonl" "$OUT/replica1/probe.jsonl" \
    "${ledgers[0]}" "${ledgers[1]}" >"$OUT/line-counts.txt"
du -h "$OUT/replica0/probe.jsonl" "$OUT/replica1/probe.jsonl" \
    "${ledgers[0]}" "${ledgers[1]}" >"$OUT/file-sizes.txt"
write_manifest
ln -sfn "$OUT" "$ROOT/raw/latest-$MODE"
trap - EXIT INT TERM
rm -r -- "$secret_dir"
secret_dir=
echo "GATEWAY_RUN_PASS mode=$MODE timestamp=$(date -u +%FT%TZ) out=$OUT"
