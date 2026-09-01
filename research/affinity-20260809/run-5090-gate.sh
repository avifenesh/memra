#!/usr/bin/env bash
# Plain-session affinity replay gate on the local RTX 5090 Laptop (lane/plain-affinity).
#
# Reproduces the deployed PP-2 plain-decode policy on a single card with MEMRA_SPEC_K=0 (the
# exact K=0 the PP-2 placement selects — see research/cachespec-20260809/RESULTS.md §hypothesis:
# the deployed slowdown is in the PLAIN path). Drives the frozen 12-turn rewritten-history pi
# workload (research/cachespec-20260809/replay.py) three times per arm and asserts, via
# compare_gate.py:
#   - EXACTNESS: every turn byte-identical between affinity ON and OFF,
#   - BUDGET:    completion_tokens <= max_tokens,
#   - SLOPE:     the ON arm's TTFT collapses after the learning turns with plain_affinity_rewinds>0.
#
# Greedy (temperature 0) makes generation deterministic, so the on/off text compare is a true
# byte-identity test of the resume path, not a sampling coincidence. MEMRA_CTX honors the 256k
# serving doctrine by default; override MEMRA_CTX for a faster smoke on the laptop's 24 GB.
set -uo pipefail

export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
DRAFT=${DRAFT:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}
PORT=${PORT:-18231}
BASE=http://127.0.0.1:$PORT
KEY=${KEY:-affinity-20260809}
TS=${AFFINITY_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${AFFINITY_OUT:-$REPO/research/affinity-20260809/raw/5090/$TS}
WORKLOAD=${WORKLOAD:-$OUT/workload.json}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)}
# 5090 laptop has 24 GB — a 256k ctx q9 session does not fit alongside the reuse pool, so the
# gate defaults to an 8k smoke ctx (the mechanism is ctx-independent: the boundary/rewind logic
# is identical). Set MEMRA_CTX=262144 to run the doctrine ctx on a bigger card (box1/RunPod).
CTX=${MEMRA_CTX:-8192}
SEQUENTIAL=${SEQUENTIAL:-12}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-256}
BASE_NOTES=${BASE_NOTES:-8}
CPU_QUOTA=${CPU_QUOTA:-600%}   # keep the desktop responsive (global rule: no uncapped local jobs)
SERVER_PID=

mkdir -p "$OUT"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}
trap stop_server EXIT INT TERM

wait_up() {
    local pid=$1
    for _ in $(seq 1 360); do
        curl -sf -H "Authorization: Bearer $KEY" "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

# systemd-run --scope keeps the CPU quota (global rule); falls back to a bare launch if not root.
launch() {
    local runner=()
    if command -v systemd-run >/dev/null 2>&1 && systemd-run --scope --quiet true 2>/dev/null; then
        runner=(systemd-run --scope -p CPUQuota="$CPU_QUOTA" --quiet)
    fi
    "${runner[@]}" env "$@" >"$LOG" 2>&1 &
    SERVER_PID=$!
}

boot_server() {
    local arm=$1 affinity=$2
    LOG=$OUT/$arm/server.log
    mkdir -p "$OUT/$arm/responses"
    launch \
        -u MEMRA_SERVE_SPEC \
        MEMRA_AFFINITY="$affinity" \
        MEMRA_SPEC_K=0 \
        MEMRA_MODELS="q9=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_CTX="$CTX" \
        MEMRA_MAX_SESSIONS="$CONCURRENCY" \
        MEMRA_REUSE_POOL="${REUSE_POOL:-4}" \
        MEMRA_PREFIX_CACHE_MB="${PREFIX_CACHE_MB:-512}" \
        MEMRA_API_KEY="$KEY" \
        MEMRA_TTFT_TRACE=1 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        "$BIN"
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: $arm server did not become ready"; tail -120 "$LOG" || true; return 1
    fi
    echo "$arm (affinity=$affinity) ready pid=$SERVER_PID $(date -u +%FT%TZ)"
}

run_arm() {
    local arm=$1 affinity=$2 mode=$3 workload=$4
    mkdir -p "$OUT/$arm"
    (
        exec 9>/tmp/memra-gpu.lock
        flock -w "${LOCK_WAIT:-3600}" 9 || { echo "FAIL: GPU lock timeout for $arm"; exit 75; }
        echo "=== arm=$arm affinity=$affinity mode=$mode $(date -u +%FT%TZ)"
        boot_server "$arm" "$affinity" || { stop_server; exit 1; }
        timeout 3600 python3 research/cachespec-20260809/replay.py \
            --base "$BASE" --model q9 --api-key "$KEY" \
            --mode "$mode" --workload "$workload" \
            --out "$OUT/$arm/requests.jsonl" --raw-dir "$OUT/$arm/responses" \
            --sequential "$SEQUENTIAL" --concurrency "$CONCURRENCY" \
            --max-tokens "$MAX_TOKENS" --base-notes "$BASE_NOTES" \
            >"$OUT/$arm/client.log" 2>&1
        local rc=$?
        curl -sf -H "Authorization: Bearer $KEY" "$BASE/metrics" \
            >"$OUT/$arm/metrics-final.json" 2>/dev/null || true
        stop_server
        echo "=== arm=$arm rc=$rc $(date -u +%FT%TZ)"
        [[ $rc -ne 0 ]] && { echo "--- client tail"; tail -60 "$OUT/$arm/client.log"; \
                             echo "--- server tail"; tail -80 "$OUT/$arm/server.log"; }
        exit "$rc"
    )
}

echo "=== plain-affinity 5090 gate ts=$TS commit=$SOURCE_COMMIT host=$(hostname)"
echo "model=$MODEL"
echo "shape: seq=$SEQUENTIAL c=$CONCURRENCY max_tokens=$MAX_TOKENS ctx=$CTX spec_k=0 (plain policy)"
test -f "$BIN"   || { echo "FAIL: missing binary $BIN (build: cargo build --release -p memra-server)"; exit 1; }
test -f "$MODEL" || { echo "FAIL: missing trunk $MODEL"; exit 1; }
test -f "$DRAFT" || { echo "FAIL: missing draft $DRAFT"; exit 1; }
sha256sum "$MODEL" "$DRAFT" >"$OUT/artifact-sha256.txt"

# RECORD the workload once on a cold affinity-ON server (its own generations freeze the workload
# prompts; the replay arms then re-issue the identical prompt bytes).
run_arm record-on 1 record "$WORKLOAD" || exit 1

# N=3 REPLAY per arm against the frozen workload.
for run in 1 2 3; do
    run_arm "on-$run"  1 replay "$WORKLOAD" || exit 1
    run_arm "off-$run" 0 replay "$WORKLOAD" || exit 1
done

echo "=== compare (N=3) $(date -u +%FT%TZ)"
rc=0
for run in 1 2 3; do
    echo "--- run $run"
    python3 research/affinity-20260809/compare_gate.py \
        --on "$OUT/on-$run/requests.jsonl" \
        --off "$OUT/off-$run/requests.jsonl" \
        --on-metrics "$OUT/on-$run/metrics-final.json" \
        --max-tokens "$MAX_TOKENS" \
        --out "$OUT/gate-run$run.json" || rc=1
done
echo "=== gate rc=$rc $(date -u +%FT%TZ)"
exit "$rc"
