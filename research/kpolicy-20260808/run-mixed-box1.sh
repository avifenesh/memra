#!/usr/bin/env bash
# Served mixed-workload comparison: pre-lane train vs request-conditioned K policy.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/kpolicy-20260808
TS=${KPOLICY_MIXED_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${KPOLICY_MIXED_OUT:-$LANE/mixed/$TS}
mkdir -p "$OUT/responses"

DRIVER=$OUT/driver.log
POINTS=$OUT/points.jsonl
SUMMARY=$OUT/SUMMARY.md
PORT=${KPOLICY_MIXED_PORT:-8151}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS=${KPOLICY_MIXED_REPS:-3}
BASELINE_COMMIT=d43f9e2707e3c19dacba993df04d467be8a3ea66
BASELINE_ROOT=${KPOLICY_BASELINE_ROOT:-$HOME/cx-kpolicy-baseline-d43f9e27}
BASELINE_TARGET=${KPOLICY_BASELINE_TARGET:-$HOME/cx-kpolicy-baseline-target}
CANDIDATE_TARGET=$(realpath -m "${CARGO_TARGET_DIR:-$ROOT/target}")

Q9=${Q9:-/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
Q9_DRAFT=${Q9_DRAFT:-/scratch-models/draft-9b-owntrim-nvfp4head-q4blk.gguf}
SHORT=$ROOT/research/e2e/prompts/p1-code-short.txt
LONG=$ROOT/research/e2e/prompts/p3-agentic-long-v3.txt
BASELINE_SERVER=$BASELINE_TARGET/release/memra-server
CANDIDATE_SERVER=$CANDIDATE_TARGET/release/memra-server

exec > >(tee "$DRIVER") 2>&1

CLEANUP_PID=
stop_server() {
    local pid=${1:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
# shellcheck disable=SC2329 # invoked through EXIT/INT/TERM traps
cleanup() {
    stop_server "$CLEANUP_PID"
}
trap cleanup EXIT INT TERM

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 240); do
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

gpu_sample() {
    {
        printf '%s,' "$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
            --format=csv,noheader | paste -sd ';' -
    } >> "$OUT/gpu.csv"
}

arm_order() {
    if (( $1 % 2 == 1 )); then
        echo "before after"
    else
        echo "after before"
    fi
}

echo "=== kpolicy mixed workload $TS ==="
echo "host=$(hostname) candidate=$(git rev-parse HEAD) baseline=$BASELINE_COMMIT"
git status --short --untracked-files=no
for artifact in "$Q9" "$Q9_DRAFT" "$SHORT" "$LONG"; do
    test -f "$artifact" || { echo "FAIL: missing artifact $artifact"; exit 1; }
done

if [[ ! -e "$BASELINE_ROOT/.git" ]]; then
    git worktree add --detach "$BASELINE_ROOT" "$BASELINE_COMMIT"
fi
test "$(git -C "$BASELINE_ROOT" rev-parse HEAD)" = "$BASELINE_COMMIT" || {
    echo "FAIL: baseline worktree is not $BASELINE_COMMIT"
    exit 1
}

echo "=== baseline build ==="
(
    cd "$BASELINE_ROOT"
    CARGO_TARGET_DIR="$BASELINE_TARGET" cargo build --release -p memra-server
) > "$OUT/build-before.log" 2>&1
cat "$OUT/build-before.log"

echo "=== candidate build ==="
CARGO_TARGET_DIR="$CANDIDATE_TARGET" cargo build --release -p memra-server \
    > "$OUT/build-after.log" 2>&1
cat "$OUT/build-after.log"

sha256sum "$BASELINE_SERVER" "$CANDIDATE_SERVER" > "$OUT/binary-sha256.txt"
sha256sum "$Q9" "$Q9_DRAFT" > "$OUT/artifact-sha256.txt"

exec 9>/tmp/memra-gpu.lock
flock -w "${KPOLICY_MIXED_LOCK_WAIT:-14400}" 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
echo "GPU lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true
gpu_sample

for rep in $(seq 1 "$REPS"); do
    for arm in $(arm_order "$rep"); do
        case "$arm" in
            before) server=$BASELINE_SERVER ;;
            after) server=$CANDIDATE_SERVER ;;
            *) echo "FAIL: unknown arm $arm"; exit 1 ;;
        esac
        label="${arm}-r${rep}"
        server_log="$OUT/${label}-server.log"
        port_free || {
            echo "FAIL: port $PORT already in use before $label"
            ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
            exit 1
        }
        echo "=== arm $label ==="
        env \
            -u MEMRA_SPEC_K \
            -u MEMRA_PP_STAGES \
            -u MEMRA_PP_DEVICES \
            -u MEMRA_PP_SHARD \
            -u MEMRA_SERVE_SPEC \
            -u MEMRA_SPEC_GATE \
            -u MEMRA_SPEC_GATE_LOW \
            -u MEMRA_SPEC_GATE_HIGH \
            CUDA_VISIBLE_DEVICES=0 \
            MEMRA_MODELS="q9=${Q9}+${Q9_DRAFT}" \
            MEMRA_ADDR="$ADDR" \
            MEMRA_COMPAT=openai \
            MEMRA_CTX=8192 \
            MEMRA_MAX_SESSIONS=8 \
            MEMRA_REUSE_POOL=4 \
            "$server" > "$server_log" 2>&1 &
        CLEANUP_PID=$!
        if ! wait_up "$CLEANUP_PID"; then
            echo "FAIL: $label server did not become ready"
            tail -100 "$server_log" || true
            exit 1
        fi
        if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
                | grep -q "pid=$CLEANUP_PID,"; then
            echo "FAIL: $label responder is not child pid $CLEANUP_PID"
            exit 1
        fi

        client_log="$OUT/${label}-client.log"
        python3 "$LANE/mixed_workload.py" \
            --base "$BASE" \
            --model q9 \
            --short "$SHORT" \
            --long "$LONG" \
            --arm "$arm" \
            --rep "$rep" \
            --out "$POINTS" \
            --raw-dir "$OUT/responses" > "$client_log" 2>&1
        cat "$client_log"

        if [[ "$arm" == after ]]; then
            for source in cold-short cold-long cached-long concurrency; do
                if ! grep -q "source=$source" "$server_log"; then
                    echo "FAIL: $label did not exercise policy source=$source"
                    exit 1
                fi
            done
        fi
        curl -sf "$BASE/metrics" > "$OUT/${label}-metrics.json" 2>&1 || true
        stop_server "$CLEANUP_PID"
        CLEANUP_PID=
        sleep 3
        gpu_sample
    done
done

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
gpu_sample
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

python3 "$LANE/analyze_mixed.py" "$POINTS" --reps "$REPS" --out "$SUMMARY"
echo "KPOLICY_MIXED_DONE out=$OUT"
