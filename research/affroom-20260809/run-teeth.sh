#!/usr/bin/env bash
# Short-window plain-affinity exactness gate: resumed checkpoint state vs every-tier cold.
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
STAMP=${AFFROOM_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${AFFROOM_OUT:-$REPO/research/affroom-20260809/raw/affinity-teeth-after-$STAMP}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
DRAFT=${DRAFT:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}
PORT=${PORT:-18232}
BASE=http://127.0.0.1:$PORT
KEY=${KEY:-affroom-teeth-20260809}
WORKLOAD=$OUT/workload.json
CPU_QUOTA=${CPU_QUOTA:-600%}

mkdir -p "$OUT"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

printf 'commit=%s\nbinary_sha256=%s\n' \
    "$(git rev-parse HEAD)" "$(sha256sum "$BIN" | cut -d' ' -f1)" > "$OUT/build.txt"
sha256sum "$MODEL" "$DRAFT" > "$OUT/artifact-sha256.txt"

(
    while :; do
        date -u +%Y-%m-%dT%H:%M:%SZ
        nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
        sleep 2
    done
) > "$OUT/gpu-samples.log" 2>&1 &
SAMPLE_PID=$!
cleanup_sampler() {
    kill "$SAMPLE_PID" 2>/dev/null || true
    wait "$SAMPLE_PID" 2>/dev/null || true
}
trap cleanup_sampler EXIT INT TERM

wait_up() {
    local pid=$1
    for _ in $(seq 1 180); do
        curl -sf -H "Authorization: Bearer $KEY" "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

run_arm() {
    local arm=$1 mode=$2 kv_reuse=$3 affinity=$4
    mkdir -p "$OUT/$arm/responses"
    (
        exec 9>/tmp/memra-gpu.lock
        flock -w 3600 9 || { echo "FAIL: GPU lock timeout for $arm"; exit 75; }
        local server_pid=
        stop_server() {
            if [[ -n ${server_pid:-} ]]; then
                kill "$server_pid" 2>/dev/null || true
                for _ in $(seq 1 60); do
                    kill -0 "$server_pid" 2>/dev/null || break
                    sleep 1
                done
                kill -9 "$server_pid" 2>/dev/null || true
                wait "$server_pid" 2>/dev/null || true
            fi
        }
        trap stop_server EXIT INT TERM

        local runner=()
        if command -v systemd-run >/dev/null 2>&1 \
            && systemd-run --scope --quiet true 2>/dev/null; then
            runner=(systemd-run --scope -p CPUQuota="$CPU_QUOTA" --quiet)
        fi
        echo "arm=$arm mode=$mode kv_reuse=$kv_reuse affinity=$affinity start=$(date -u +%FT%TZ)"
        "${runner[@]}" env -u MEMRA_SERVE_SPEC \
            MEMRA_KV_REUSE="$kv_reuse" \
            MEMRA_AFFINITY="$affinity" \
            MEMRA_SPEC_K=0 \
            MEMRA_MODELS="q9=$MODEL+$DRAFT" \
            MEMRA_COMPAT=openai \
            MEMRA_CTX=8192 \
            MEMRA_MAX_SESSIONS=4 \
            MEMRA_REUSE_POOL=4 \
            MEMRA_PREFIX_CACHE_MB=0 \
            MEMRA_API_KEY="$KEY" \
            MEMRA_TTFT_TRACE=1 \
            MEMRA_ADDR="127.0.0.1:$PORT" \
            "$BIN" > "$OUT/$arm/server.log" 2>&1 &
        server_pid=$!
        if ! wait_up "$server_pid"; then
            echo "FAIL: $arm server did not become ready"
            tail -100 "$OUT/$arm/server.log" || true
            exit 1
        fi

        timeout 3600 python3 research/cachespec-20260809/replay.py \
            --base "$BASE" --model q9 --api-key "$KEY" \
            --mode "$mode" --workload "$WORKLOAD" \
            --out "$OUT/$arm/requests.jsonl" --raw-dir "$OUT/$arm/responses" \
            --sequential 12 --concurrency 4 --max-tokens 8 --base-notes 8 \
            > "$OUT/$arm/client.log" 2>&1
        local rc=$?
        curl -sf -H "Authorization: Bearer $KEY" "$BASE/metrics" \
            > "$OUT/$arm/metrics-final.json" 2>/dev/null || true
        stop_server
        server_pid=
        echo "arm=$arm rc=$rc complete=$(date -u +%FT%TZ)"
        if [[ $rc -ne 0 ]]; then
            tail -80 "$OUT/$arm/client.log" || true
        fi
        exit "$rc"
    )
}

run_arm record record 1 1 || exit 1
run_arm aff-only replay 1 1 || exit 1
run_arm cold-iso replay 0 0 || exit 1

set +e
python3 research/affinity-20260809/compare_gate.py \
    --on "$OUT/aff-only/requests.jsonl" \
    --cold "$OUT/cold-iso/requests.jsonl" \
    --short-on "$OUT/aff-only/requests.jsonl" \
    --short-cold "$OUT/cold-iso/requests.jsonl" \
    --on-metrics "$OUT/aff-only/metrics-final.json" \
    --max-tokens 8 \
    --out "$OUT/gate-teeth.json" 2>&1 | tee "$OUT/compare.log"
rc=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\n' "$rc" > "$OUT/verdict.txt"
exit "$rc"
