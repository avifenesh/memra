#!/usr/bin/env bash
# One bounded box1 lock block for the increment-1 speculative-pipeline exactness gates.
set -euo pipefail

ROOT=${SPECMECH_ROOT:-/home/ubuntu/memra-specmech}
OUT=${SPECMECH_OUT:-/home/ubuntu/specmech-receipts/exactness}
PORT=${SPECMECH_PORT:-8142}
ADDR=127.0.0.1:${PORT}
BASE=http://${ADDR}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
GOLDEN=/home/ubuntu/darktrain2/golden-response.bin
SERVER=${ROOT}/target/release/memra-server
RUN_SPEC=${ROOT}/target/release/run-spec
QOS=${ROOT}/research/p0iso-20260810/qos_probe.py
RUNSPEC_GATE=${SPECMECH_RUNSPEC_GATE:-1}
SERVE_ARMS=${SPECMECH_SERVE_ARMS:-"plain serial pipe"}
ALLOW_MISMATCH=${SPECMECH_ALLOW_MISMATCH:-0}
REQUESTS=${SPECMECH_REQUESTS:-2}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$RUN_SPEC" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } > "$path" 2>&1
}

wait_up() {
    local pid=$1 log=$2
    for _ in $(seq 1 450); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || { tail -100 "$log"; return 1; }
        sleep 2
    done
    tail -100 "$log"
    return 1
}

server_pid=
stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|illegal|sentinel|spec pending flush failed' \
        "$log" || true)
    if [[ -n "$failures" ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_serve_arm() {
    local arm=$1 dir="$OUT/$1" log="$OUT/$1/server.log"
    local -a policy
    mkdir -p "$dir"
    case "$arm" in
        plain) policy=(MEMRA_SERVE_SPEC=0) ;;
        serial-host) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1) ;;
        serial) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1) ;;
        pipe) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_SPEC_PIPE=1) ;;
        *) echo "FAIL: unknown arm $arm"; return 1 ;;
    esac

    echo "=== $arm $(date -u +%FT%TZ) ==="
    env \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_STATS \
        -u MEMRA_SPEC_DEVACC \
        -u MEMRA_SPEC_PIPE \
        -u MEMRA_SPEC_PP_ANATOMY \
        -u MEMRA_SPEC_REPLAY \
        -u MEMRA_SPEC_STREAM \
        -u MEMRA_SPEC_ADAPT \
        -u MEMRA_SPEC_PMIN0 \
        -u MEMRA_SPEC_PMIN \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PREFIX_DEDUP=1 \
        MEMRA_PRIME_BATCH_HOLD_MS=4 \
        MEMRA_MODELS="step37=${MODEL}+${DRAFT}" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="$ADDR" \
        "$SERVER" > "$log" 2>&1 &
    server_pid=$!
    wait_up "$server_pid" "$log"
    snapshot "$dir/serve-ready.log" "$arm-ready"
    local probe_rc
    set +e
    "$QOS" \
        --base "$BASE" \
        --model step37 \
        --label "$arm" \
        --requests "$REQUESTS" \
        --max-tokens 64 \
        --golden "$GOLDEN" \
        --rows "$dir/qos-rows.jsonl" \
        --summary "$dir/qos-summary.json"
    probe_rc=$?
    set -e
    echo "$probe_rc" > "$dir/probe-exit-code.txt"
    curl -sf "$BASE/metrics" > "$dir/metrics-after.json"
    sleep 1
    stop_server
    assert_server_clean "$log"

    local spec_lines pipe_lines
    spec_lines=$(grep -c '\[spec-acc\]' "$log" || true)
    pipe_lines=$(grep -c '\[spec-pipe\]' "$log" || true)
    echo "arm=$arm spec_lines=$spec_lines pipe_lines=$pipe_lines"
    if [[ "$arm" == plain ]]; then
        test "$spec_lines" -eq 0
        test "$pipe_lines" -eq 0
    elif [[ "$arm" == pipe ]]; then
        test "$spec_lines" -gt 0
        test "$pipe_lines" -gt 0
    else
        test "$spec_lines" -gt 0
        test "$pipe_lines" -eq 0
    fi
    if [[ "$probe_rc" -ne 0 && "$ALLOW_MISMATCH" -ne 1 ]]; then
        return "$probe_rc"
    fi
}

run_locked() {
    local apps pass_count k_count
    echo "EXACTNESS_LOCK_ACQUIRED $(date -u +%FT%TZ)"
    echo "host=$(hostname)"
    echo "source_commit=$(git rev-parse HEAD)"
    git status --short --branch
    sha256sum "$RUN_SPEC" "$SERVER" "$MODEL" "$DRAFT" "$GOLDEN" "$QOS" > "$OUT/SHA256SUMS"
    stat -c '%n %s bytes %y' "$MODEL" "$DRAFT" "$GOLDEN" > "$OUT/artifacts.txt"
    snapshot "$OUT/nvidia-smi-before.log" lock-acquired
    apps=$(compute_apps 2>/dev/null || true)
    test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; return 1; }

    if [[ "$RUNSPEC_GATE" -eq 1 ]]; then
        echo "=== run-spec K=1..8 $(date -u +%FT%TZ) ==="
        env \
        -u MEMRA_PROMPT_DIR \
        -u MEMRA_SPEC_K \
        -u MEMRA_GEN_ONLY \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_SPEC_TEMP=0 \
        MEMRA_SPEC_DEVACC=1 \
        MEMRA_SPEC_PIPE=1 \
        MEMRA_MTP_DRAFT="$DRAFT" \
        MEMRA_NGEN=32 \
        MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
            timeout 1800 "$RUN_SPEC" "$MODEL" 2>&1 | tee "$OUT/run-spec.log"
        pass_count=$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log" || true)
        k_count=$(grep -c '^\[generate_spec K=' "$OUT/run-spec.log" || true)
        test "$pass_count" -eq 8
        test "$k_count" -eq 8
        grep -q '^=== SELF-CONSISTENCY PASS ===$' "$OUT/run-spec.log"
    fi

    local arm
    for arm in $SERVE_ARMS; do
        run_serve_arm "$arm"
    done

    if [[ "$SERVE_ARMS" == "plain serial pipe" && "$ALLOW_MISMATCH" -eq 0 \
          && "$REQUESTS" -eq 2 ]]; then
        python3 - "$OUT/plain/qos-rows.jsonl" "$OUT/serial/qos-rows.jsonl" \
            "$OUT/pipe/qos-rows.jsonl" "$GOLDEN" "$OUT/identity-summary.json" <<'PY'
import base64
import hashlib
import json
import pathlib
import sys

plain_path, serial_path, pipe_path, golden_path, out_path = map(pathlib.Path, sys.argv[1:])
golden = golden_path.read_bytes()

def completions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(row.get("ok") and row.get("golden_match") for row in rows)
    return [base64.b64decode(row["text_utf8_b64"]) for row in rows]

plain = completions(plain_path)
serial = completions(serial_path)
pipe = completions(pipe_path)
assert all(value == golden for value in plain + serial + pipe)
summary = {
    "plain_requests": len(plain),
    "serial_requests": len(serial),
    "pipe_requests": len(pipe),
    "plain_vs_serial_vs_pipe": "byte-identical",
    "golden_bytes": len(golden),
    "golden_sha256": hashlib.sha256(golden).hexdigest(),
    "unique_completion_sha256": sorted(
        {hashlib.sha256(v).hexdigest() for v in plain + serial + pipe}
    ),
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
    fi

    snapshot "$OUT/nvidia-smi-after.log" exactness-complete
    apps=$(compute_apps 2>/dev/null || true)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; return 1; }
    echo "EXACTNESS_PASS $(date -u +%FT%TZ)"
}

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
run_locked
flock -u 9
