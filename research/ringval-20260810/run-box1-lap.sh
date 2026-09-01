#!/usr/bin/env bash
# Deliberately lap a plain-affinity checkpoint, require decline, then compare cold output.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
BIN=${BIN:-$TARGET/release/memra-server}
TOK_CHECK=${TOK_CHECK:-$TARGET/release/tok-check}
PROBE=${PROBE:-$(cd "$(dirname "$0")" && pwd)/lap_probe.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18434}
BASE=http://127.0.0.1:$PORT
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/lap-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_BINARY=${EXPECTED_BINARY:-7f04f76715d637c46a379366a833d518aed9d465a5dcfd1ffee53be79d9b9cef}
SERVER_PID=0

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

stop_server() {
    if (( SERVER_PID > 0 )); then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=0
    fi
    for _ in $(seq 1 180); do
        [[ -z $(compute_apps) ]] && return 0
        sleep 1
    done
    compute_apps
    return 1
}

cleanup() {
    stop_server || true
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -120 "$log"; return 1; }
        sleep 1
    done
    tail -120 "$log"
    return 1
}

preflight() {
    local source binary apps
    source=$(git -C "$REPO" rev-parse HEAD)
    binary=$(sha256sum "$BIN" | awk '{print $1}')
    echo "source_commit=$source"
    echo "binary_sha256=$binary"
    echo "tok_check_sha256=$(sha256sum "$TOK_CHECK" | awk '{print $1}')"
    echo "probe_sha256=$(sha256sum "$PROBE" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $binary == "$EXPECTED_BINARY" ]]
    [[ -x $TOK_CHECK && -f $PROBE ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

resolve_control_id() {
    "$TOK_CHECK" "$MODEL" '<|im_start|>' | tee "$OUT/tok-check-im-start.log"
    python3 - "$OUT/tok-check-im-start.log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r'encode\(.*?\) = \[([^]]+)\]', text)
assert match, text
ids = [int(part.strip()) for part in match.group(1).split(',')]
assert len(ids) >= 2, ids
print(ids[-1])
PY
}

start_server() {
    local label=$1 log=$OUT/$1-server.log
    env -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_API_KEY \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_SWA_RING=1 \
        MEMRA_SERVE_SPEC=0 \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_CTX=262144 \
        MEMRA_REUSE_POOL=2 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$log"
    grep -q '\[admission\].*capped at 4639 rows' "$log"
}

assert_clean() {
    local log=$1
    if grep -Ein 'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died' "$log"; then
        return 1
    fi
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight
    CONTROL_ID=$(resolve_control_id | tail -1)
    echo "im_start_control_id=$CONTROL_ID"

    start_server cold
    python3 "$PROBE" cold --base "$BASE" --control-id "$CONTROL_ID" --out "$OUT/cold"
    snapshot "$OUT/cold-before-stop.log" cold-complete
    stop_server
    assert_clean "$OUT/cold-server.log"

    start_server lap
    python3 "$PROBE" lap --base "$BASE" --control-id "$CONTROL_ID" --out "$OUT/lap"
    snapshot "$OUT/lap-before-stop.log" lap-complete
    stop_server
    assert_clean "$OUT/lap-server.log"
    grep -F '[worker] plain-affinity: declined (SWA ring lapped checkpoint 1024' \
        "$OUT/lap-server.log" >"$OUT/lap-decline-line.txt"
    grep -F '[prefix-cache] refused for MEMRA_SWA_RING=1 Step35 session' \
        "$OUT/lap-server.log" >"$OUT/prefix-refusal-lines.txt"

    cmp "$OUT/cold/cold-text.bin" "$OUT/lap/resume-declined-text.bin"
    {
        echo 'n=1'
        echo 'lap_decline_line=PASS'
        echo 'cold_reprime_full_prompt_tokens=2048'
        echo 'cold_reprime_cached_tokens=0'
        echo 'cold_reprime_output_identity=PASS'
        sha256sum "$OUT/cold/cold-text.bin" "$OUT/lap/resume-declined-text.bin"
        cat "$OUT/lap-decline-line.txt"
    } | tee "$OUT/verdict.txt"
    snapshot "$OUT/nvidia-smi-after.log" final
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
