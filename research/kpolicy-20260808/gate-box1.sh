#!/usr/bin/env bash
# Final request-conditioned K-policy battery on box1. Build outside the GPU window,
# then hold one flock across run-spec, live policy assertions, accept-gate, and serve-smoke.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/kpolicy-20260808
TS=${KPOLICY_GATE_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${KPOLICY_GATE_OUT:-$LANE/gates/$TS}
mkdir -p "$OUT"

DRIVER=$OUT/driver.log
SUMMARY=$OUT/SUMMARY.md
PORT=${KPOLICY_GATE_PORT:-8147}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
SHORT_K=${KPOLICY_SHORT_K:-3}
COLD_LONG_K=${KPOLICY_COLD_LONG_K:-3}
CACHED_LONG_K=${KPOLICY_CACHED_LONG_K:-2}

Q9=${Q9:-/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
Q9_DRAFT=${Q9_DRAFT:-/scratch-models/draft-9b-owntrim-nvfp4head-q4blk.gguf}
Q27=${Q27:-/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
Q27_DRAFT=${Q27_DRAFT:-/scratch-models/draft-daily-owntrim-nvfp4head-q4blk.gguf}
SHORT=$ROOT/research/e2e/prompts/p1-code-short.txt
LONG=$ROOT/research/e2e/prompts/p3-agentic-long-v3.txt
TARGET_DIR=$(realpath -m "${CARGO_TARGET_DIR:-$ROOT/target}")
SERVER=$TARGET_DIR/release/memra-server
RUNSPEC=$TARGET_DIR/release/run-spec
ACCEPT_CELLS=$OUT/accept-cells-box1.tsv

FAILS=0
CLEANUP_PID=
TARGET_LINK_CREATED=0

exec > >(tee "$DRIVER") 2>&1

pass() {
    echo "PASS: $*"
}

fail() {
    echo "FAIL: $*"
    FAILS=$((FAILS + 1))
}

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
    if ((TARGET_LINK_CREATED)); then
        unlink "$ROOT/target"
    fi
}
trap cleanup EXIT INT TERM

run_capture() {
    local log=$1
    shift
    echo "=== run: $log ==="
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" > "$log" 2>&1
    local rc=$?
    set -e
    cat "$log"
    return "$rc"
}

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${1}[[:space:]]"
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

wait_worker_idle() {
    for _ in $(seq 1 120); do
        if curl -sf "$BASE/metrics" 2>/dev/null | python3 -c '
import json
import sys
try:
    metrics = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if metrics.get("serve_idle_seconds", 0.0) >= 0.5 else 1)
'; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

launch_server() {
    local label=$1
    shift
    local log=$OUT/$label-server.log
    local -a command=(
        env
        -u CUDA_VISIBLE_DEVICES
        -u MEMRA_PP_STAGES
        -u MEMRA_PP_DEVICES
        -u MEMRA_PP_SHARD
        -u MEMRA_SERVE_SPEC
        -u MEMRA_SPEC_GATE
        -u MEMRA_SPEC_GATE_LOW
        -u MEMRA_SPEC_GATE_HIGH
        -u MEMRA_SPEC_K
        -u MEMRA_API_KEY
        -u MEMRA_API_KEYS
        "$@"
        MEMRA_MODELS="q9=${Q9}+${Q9_DRAFT}"
        MEMRA_ADDR="$ADDR"
        MEMRA_COMPAT=openai
        MEMRA_CTX=8192
        MEMRA_MAX_SESSIONS=8
        MEMRA_REUSE_POOL=4
        "$SERVER"
    )

    if ! port_free "$PORT"; then
        fail "port $PORT already in use before $label"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    fi

    echo "=== server: $label ==="
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}" > "$log" 2>&1 &
    CLEANUP_PID=$!
    if ! wait_up "$CLEANUP_PID"; then
        fail "$label did not become ready"
        tail -100 "$log" || true
        stop_server "$CLEANUP_PID"
        CLEANUP_PID=
        return 1
    fi
    if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
            | grep -q "pid=$CLEANUP_PID,"; then
        fail "$label responder is not child pid $CLEANUP_PID"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        stop_server "$CLEANUP_PID"
        CLEANUP_PID=
        return 1
    fi
}

finish_server() {
    local label=$1
    if ! wait_worker_idle; then
        fail "$label worker did not quiesce before shutdown"
    fi
    stop_server "$CLEANUP_PID"
    CLEANUP_PID=
    sleep 3
}

policy_cell() {
    local label=$1 prompt_class=$2 prompt=$3 expected_k=$4
    local log=$OUT/$label-client.log
    if run_capture "$log" \
            python3 "$LANE/measure_client.py" \
                --base "$BASE" \
                --model q9 \
                --class "$prompt_class" \
                --prompt "$prompt" \
                --k "$expected_k" \
                --rep 1 \
                --max-tokens 128 \
                --out "$OUT/policy-points.jsonl" \
                --raw-dir "$OUT/responses"; then
        pass "$label response selected K=$expected_k"
    else
        fail "$label response"
        return 1
    fi
}

load_c4() {
    local label=$1
    if run_capture "$OUT/$label-load.log" \
            python3 tools/load-serve.py \
                --base "$BASE" \
                --model q9 \
                --concurrency 4 \
                --requests 4 \
                --max-tokens 256 \
                --greedy \
                --warmup 0 \
                --label "$label" \
                --out "$OUT/load-points.jsonl"; then
        pass "$label load completed"
    else
        fail "$label load"
    fi
}

check_runspec() {
    local log=$1
    local passes ks
    passes=$(grep -c "self-consistency: PASS" "$log" || true)
    ks=$(grep -cE '^\[generate_spec K=[1-8]\]' "$log" || true)
    if [[ "$passes" -eq 8 && "$ks" -eq 8 ]] \
            && grep -q "=== SELF-CONSISTENCY PASS ===" "$log" \
            && ! grep -q "self-consistency: FAIL" "$log"; then
        pass "run-spec K=1..8"
    else
        fail "run-spec K=1..8 (rows=$ks passes=$passes)"
    fi
}

for artifact in "$Q9" "$Q9_DRAFT" "$Q27" "$Q27_DRAFT" "$SHORT" "$LONG"; do
    test -f "$artifact" || {
        echo "FAIL: missing artifact $artifact"
        exit 1
    }
done

echo "=== kpolicy gate $TS ==="
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
git status --short --untracked-files=no
sha256sum "$Q9" "$Q9_DRAFT" "$Q27" "$Q27_DRAFT" > "$OUT/artifact-sha256.txt"

echo "=== release build ==="
{
    cargo build --release -p memra-server
    cargo build --release -p memra-engine --bin run-spec
} > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
sha256sum "$SERVER" "$RUNSPEC" > "$OUT/binary-sha256.txt"

if [[ "$TARGET_DIR" != "$ROOT/target" ]]; then
    if [[ -e "$ROOT/target" || -L "$ROOT/target" ]]; then
        if [[ "$(realpath -m "$ROOT/target")" != "$TARGET_DIR" ]]; then
            echo "FAIL: $ROOT/target exists but does not resolve to $TARGET_DIR"
            exit 1
        fi
    else
        ln -s "$TARGET_DIR" "$ROOT/target"
        TARGET_LINK_CREATED=1
    fi
fi

python3 - "$ACCEPT_CELLS" "$Q9" "$Q9_DRAFT" "$Q27" "$Q27_DRAFT" <<'PY'
import sys

out, q9, q9_draft, q27, q27_draft = sys.argv[1:]
rows = []
for line in open("tools/fast-gate/accept-cells.tsv"):
    if line.startswith("#") or not line.strip():
        rows.append(line)
        continue
    fields = line.rstrip("\n").split("\t")
    if fields[0].startswith("q9-"):
        fields[1:3] = [q9, q9_draft]
    elif fields[0].startswith("q27-"):
        fields[1:3] = [q27, q27_draft]
    rows.append("\t".join(fields) + "\n")
open(out, "w").writelines(rows)
PY

exec 9>/tmp/memra-gpu.lock
flock -w "${KPOLICY_GATE_LOCK_WAIT:-14400}" 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
echo "GPU lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true

echo "=== run-spec exactness ==="
if run_capture "$OUT/run-spec.log" \
        env \
            -u MEMRA_PP_STAGES \
            -u MEMRA_PP_DEVICES \
            -u MEMRA_SPEC_K \
            CUDA_VISIBLE_DEVICES=0 \
            MEMRA_QWEN_DC=0 \
            MEMRA_MTP_DRAFT="$Q9_DRAFT" \
            MEMRA_SPEC_TEMP=0 \
            MEMRA_NGEN=64 \
            "$RUNSPEC" "$Q9" 55; then
    check_runspec "$OUT/run-spec.log"
else
    fail "run-spec command"
fi

echo "=== automatic prompt/cache table ==="
if launch_server policy-table CUDA_VISIBLE_DEVICES=0; then
    policy_cell policy-short cold-short "$SHORT" "$SHORT_K" || true
    policy_cell policy-cold-long cold-long "$LONG" "$COLD_LONG_K" || true
    policy_cell policy-cached-long cached-long "$LONG" "$CACHED_LONG_K" || true
    finish_server policy-table
    table_log=$OUT/policy-table-server.log
    for check in \
        "K=${SHORT_K} source=cold-short" \
        "K=${COLD_LONG_K} source=cold-long" \
        "K=${CACHED_LONG_K} source=cached-long"; do
        if grep -Eq "\\[spec-k\\].*${check}" "$table_log"; then
            pass "policy log $check"
        else
            fail "policy log $check"
        fi
    done
fi

echo "=== PP-2 placement row ==="
if launch_server policy-pp2 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1; then
    policy_cell policy-pp2-short cold-short "$SHORT" 0 || true
    finish_server policy-pp2
    pp2_log=$OUT/policy-pp2-server.log
    if grep -Eq '\[spec-k\].*K=0 source=pp2-placement' "$pp2_log"; then
        pass "PP-2 resolves to K=0"
    else
        fail "PP-2 K=0 policy log"
    fi
    if grep -q '\[spec-acc\]' "$pp2_log"; then
        fail "PP-2 automatic row ran spec"
    else
        pass "PP-2 automatic row stayed plain"
    fi
fi

echo "=== operator pin precedence ==="
if launch_server policy-pin-pp2 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_SPEC_K="$COLD_LONG_K"; then
    policy_cell policy-pin-pp2-short cold-short "$SHORT" "$COLD_LONG_K" || true
    finish_server policy-pin-pp2
    pin_log=$OUT/policy-pin-pp2-server.log
    if grep -Eq "\\[spec-k\\].*K=${COLD_LONG_K} source=operator-pin" "$pin_log" \
            && grep -q '\[spec-acc\]' "$pin_log"; then
        pass "MEMRA_SPEC_K pin overrides PP-2 automatic K=0"
    else
        fail "MEMRA_SPEC_K PP-2 precedence"
    fi
fi

echo "=== #89 single-card crossover ==="
if launch_server policy-c4 CUDA_VISIBLE_DEVICES=0; then
    load_c4 policy-c4
    finish_server policy-c4
    c4_log=$OUT/policy-c4-server.log
    if grep -Eq "\\[spec-k\\].*K=${SHORT_K} source=cold-short" "$c4_log"; then
        pass "c=4 first wave admitted positive-K requests"
    else
        fail "c=4 positive-K admission"
    fi
    if grep -Eq '\[spec-k\].*K=0 source=concurrency' "$c4_log"; then
        pass "c=4 overflow arrivals resolved to K=0"
    else
        fail "c=4 K=0 concurrency row"
    fi
    if grep -q '\[spec-gate\] demoted session to batched decode' "$c4_log"; then
        pass "c=4 live-session demotion preserved"
    else
        fail "c=4 live-session demotion"
    fi
fi

echo "=== served acceptance gate ==="
if port_free 8317; then
    if run_capture "$OUT/accept-gate.log" \
            env \
                -u MEMRA_SPEC_K \
                -u MEMRA_PP_STAGES \
                -u MEMRA_PP_DEVICES \
                CUDA_VISIBLE_DEVICES=0 \
                MEMRA_ACCEPT_CELLS_TSV="$ACCEPT_CELLS" \
                MEMRA_ACCEPT_LOGDIR="$OUT/accept" \
                MEMRA_ACCEPT_PORT=8317 \
                tools/accept-gate.sh; then
        pass "accept-gate"
    else
        fail "accept-gate"
    fi
else
    fail "accept-gate port 8317 already in use"
fi

echo "=== repository serve-smoke ==="
if port_free 8177; then
    if run_capture "$OUT/serve-smoke.log" \
            env \
                -u MEMRA_SPEC_K \
                -u MEMRA_PP_STAGES \
                -u MEMRA_PP_DEVICES \
                -u MEMRA_SERVE_SPEC \
                -u MEMRA_SPEC_GATE \
                -u MEMRA_SPEC_GATE_LOW \
                -u MEMRA_SPEC_GATE_HIGH \
                CUDA_VISIBLE_DEVICES=0 \
                tools/serve-smoke.sh "$Q9" "$Q9_DRAFT"; then
        pass "tools/serve-smoke.sh"
    else
        fail "tools/serve-smoke.sh"
    fi
    test ! -f /tmp/serve-smoke.log \
        || cp /tmp/serve-smoke.log "$OUT/serve-smoke-last-server.log"
else
    fail "serve-smoke port 8177 already in use"
fi

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post.csv"
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

{
    echo "# K-policy gate summary"
    echo
    echo "- Host: $(hostname)"
    echo "- Commit: $(git rev-parse HEAD)"
    echo "- Script-detected failures: $FAILS"
    echo
    echo "## Policy decisions"
    grep -H '\[spec-k\].*source=' "$OUT"/policy-*-server.log \
        | sed "s|$OUT/||" || true
    echo
    echo "## Gate verdicts"
    grep -E '^PASS:|^FAIL:|^accept-gate:|^serve-smoke:' "$DRIVER" || true
} > "$SUMMARY"
cat "$SUMMARY"

echo "KPOLICY_GATES_DONE failures=$FAILS"
exit "$FAILS"
