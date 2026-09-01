#!/usr/bin/env bash
# Single-card Q27/Q35 requalification under one uninterrupted box1 GPU lock.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${REQUAL2_ROOT:-/opt/dl-image/nvme/cx-requal2}
REPO=$ROOT/memra
HARNESS=$ROOT/harness
MODELS=${REQUAL2_MODELS:-/opt/dl-image/nvme/cx-requal/models}
EXPECTED_SOURCE=${REQUAL2_EXPECTED_SOURCE:?set REQUAL2_EXPECTED_SOURCE}
STAMP=${REQUAL2_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${REQUAL2_OUT:-$ROOT/raw/run-$STAMP}

Q27=$MODELS/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=$MODELS/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=$MODELS/draft-35b-owntrim-nvfp4head-q4blk.gguf
EXPECTED_Q27=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_Q27_DRAFT=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
EXPECTED_Q35=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
EXPECTED_Q35_DRAFT=ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a

KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
SERVER=$REPO/target/release/memra-server
PROMPT=$REPO/research/e2e/prompts/pp512.txt
PREFIX_GATE=$HARNESS/prefix_exactness.py
FROZEN_REPLAY=$HARNESS/sellgate_replay.py
WORKLOAD_LOCK=$HARNESS/workload.lock.json
SINGLE_REPLAY=$HARNESS/single_replay.py

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT/gates" "$OUT/campaign" "$OUT/exactness"
exec > >(tee "$OUT/orchestrator.log") 2>&1

server_pid=
sampler_pid=
vmstat_pid=
dmon_pid=

compute_apps_gpu0() {
    nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi -i 0 \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps_gpu0 | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

check_hash() {
    local expected=$1 path=$2 actual
    test -f "$path"
    actual=$(sha256sum "$path" | awk '{print $1}')
    echo "$actual  $path"
    test "$actual" = "$expected"
}

preflight() {
    test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
    test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
    check_hash "$EXPECTED_Q27" "$Q27"
    check_hash "$EXPECTED_Q27_DRAFT" "$Q27_DRAFT"
    check_hash "$EXPECTED_Q35" "$Q35"
    check_hash "$EXPECTED_Q35_DRAFT" "$Q35_DRAFT"
    check_hash 91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b "$FROZEN_REPLAY"
    check_hash 85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34 "$WORKLOAD_LOCK"
    python3 -m py_compile "$PREFIX_GATE" "$FROZEN_REPLAY" "$SINGLE_REPLAY"
    python3 -m json.tool "$WORKLOAD_LOCK" >/dev/null
    for binary in "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER"; do
        test -x "$binary"
    done
}

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ)"
    set +e
    "$@" >"$log" 2>&1
    local rc=$?
    set -e
    echo "RUN_DONE label=$label rc=$rc ts=$(date -u +%FT%TZ)"
    return "$rc"
}

stop_server() {
    test -n "${server_pid:-}" || return 0
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" 2>/dev/null || true
            server_pid=
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server pid=$server_pid did not stop after 120s; sending KILL"
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
    return 1
}

stop_samplers() {
    local pid
    for pid in "${sampler_pid:-}" "${vmstat_pid:-}" "${dmon_pid:-}"; do
        test -n "$pid" || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    sampler_pid=
    vmstat_pid=
    dmon_pid=
}

cleanup() {
    stop_server || true
    stop_samplers
}

wait_ready() {
    local base=$1 log=$2
    for _ in $(seq 1 900); do
        curl -sf "$base/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server never became ready"
    tail -200 "$log"
    return 1
}

assert_server_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' \
        "$log" || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

start_server() {
    local target=$1 rep=$2 run_dir=$3 model port
    case "$target" in
        q27) model=$Q27; port=18427 ;;
        q35) model=$Q35; port=18435 ;;
        *) return 2 ;;
    esac
    if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
        echo "FAIL: port $port already has a listener"
        return 1
    fi
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="$target=$model" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$port" \
        MEMRA_TAG="cx-requal2-$target-r$rep" MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$run_dir/server.log" 2>&1 &
    server_pid=$!
    wait_ready "http://127.0.0.1:$port" "$run_dir/server.log"
    curl -sf "http://127.0.0.1:$port/v1/models" >"$run_dir/models.json"
    curl -sf "http://127.0.0.1:$port/metrics" >"$run_dir/metrics-before.json"
}

run_model_rep() {
    local target=$1 rep=$2
    local run_dir
    run_dir="$OUT/campaign/r$(printf '%02d' "$rep")-$target"
    local port levels base
    mkdir -p "$run_dir"
    case "$target" in
        q27) port=18427; levels=1,2,4,8,12,16,20 ;;
        q35) port=18435; levels=1,2,4,8,16,32,40,48 ;;
        *) return 2 ;;
    esac
    base="http://127.0.0.1:$port"
    echo "MODEL_REP_START target=$target rep=$rep ts=$(date -u +%FT%TZ)"
    start_server "$target" "$rep" "$run_dir"
    snapshot "$run_dir/gpu-server-ready.log" "$target-r$rep-ready"

    if test "$rep" -eq 1; then
        run_logged "prefix-$target" "$OUT/exactness/$target.log" \
            timeout 7200 python3 "$PREFIX_GATE" \
            --endpoint "$target,$base,$target" --workload-lock "$WORKLOAD_LOCK" \
            --out "$OUT/exactness/$target.jsonl" --namespace "cx-requal2-$target-exact"
        grep -q '"verdict": "PASS"' "$OUT/exactness/$target.jsonl"
        curl -sf "$base/metrics" >"$run_dir/metrics-after-exactness.json"
    fi

    run_logged "replay-$target-r$rep" "$run_dir/replay.log" \
        timeout 21600 python3 "$SINGLE_REPLAY" \
        --endpoint "$target,$base,$target" --frozen-replay "$FROZEN_REPLAY" \
        --workload-lock "$WORKLOAD_LOCK" --out "$run_dir/replay.jsonl" \
        --namespace "cx-requal2-$target-r$rep" --target "$target" \
        --levels "$levels" --repetition "$rep" --timeout 1800 --record-failures
    local expected_cells verdict
    expected_cells=$(( $(tr -cd ',' <<<"$levels" | wc -c) * 2 + 2 ))
    verdict=$(python3 - "$run_dir/replay.jsonl" "$expected_cells" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
summary = [row for row in rows if row.get("kind") == "summary"]
assert len(summary) == 1, summary
assert summary[0].get("cells") == int(sys.argv[2]), summary
print(summary[0].get("verdict"))
PY
    )
    printf '%s\n' "$verdict" >"$run_dir/verdict.txt"
    curl -sf "$base/metrics" >"$run_dir/metrics-final.json"
    stop_server
    assert_server_clean "$run_dir/server.log"
    grep -c '^\[prime-batch\]' "$run_dir/server.log" >"$run_dir/prime-batch-count.txt" || true
    test "$(<"$run_dir/prime-batch-count.txt")" -gt 0
    test -z "$(compute_apps_gpu0)" || { compute_apps_gpu0; echo "FAIL: GPU0 process remained"; return 1; }
    echo "MODEL_REP_COMPLETE target=$target rep=$rep verdict=$verdict ts=$(date -u +%FT%TZ)"
}

write_manifest() {
    local temp
    temp=$(mktemp "$OUT/.manifest.XXXXXX")
    (
        cd "$OUT"
        find . -type f ! -name MANIFEST.sha256 ! -name orchestrator.log ! -name '.manifest.*' \
            -print0 | sort -z | xargs -0 sha256sum
    ) >"$temp"
    mv "$temp" "$OUT/MANIFEST.sha256"
}

echo "REQUAL2_START ts=$(date -u +%FT%TZ) pid=$$ source=$EXPECTED_SOURCE"
preflight | tee "$OUT/preflight.log"
{
    echo "runtime_source=$EXPECTED_SOURCE"
    echo "shape=physical GPU0 only; one model resident at a time"
    echo "model_boot_order=odd repetitions q27,q35; even repetitions q35,q27"
    echo "q27_levels=1,2,4,8,12,16,20"
    echo "q35_levels=1,2,4,8,16,32,40,48"
    echo "repetitions=5"
    echo "MEMRA_PRIME_BATCH=<unset>; naked shipping default"
    hostname
    uname -a
    git -C "$REPO" log -5 --oneline --decorate
    rustc --version
    cargo --version
    nvcc --version
    nvidia-smi -i 0 --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} >"$OUT/provenance.txt" 2>&1
sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" \
    "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" "$PROMPT" \
    "$PREFIX_GATE" "$FROZEN_REPLAY" "$WORKLOAD_LOCK" "$SINGLE_REPLAY" \
    >"$OUT/SHA256SUMS.input"

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "REQUAL2_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
trap cleanup EXIT INT TERM
snapshot "$OUT/gpu-before.log" before
test -z "$(compute_apps_gpu0)" || { compute_apps_gpu0; echo "FAIL: GPU0 not idle"; exit 1; }

nvidia-smi -i 0 \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
sampler_pid=$!
vmstat 1 >"$OUT/vmstat-1s.log" 2>&1 &
vmstat_pid=$!
nvidia-smi dmon -i 0 -s pucm -d 1 -o DT >"$OUT/pcie-dmon-1s.log" 2>&1 &
dmon_pid=$!

run_logged kernel "$OUT/gates/kernel-gpu0.log" \
    timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
grep -q 'ALL GREEN' "$OUT/gates/kernel-gpu0.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/gates/kernel-gpu0.log"; then
    echo "FAIL: kernel checker emitted a failure marker"
    exit 1
fi

run_logged run-gen-q27 "$OUT/gates/run-gen-q27.log" \
    timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q27"
grep -q 'argmax=.*MATCH' "$OUT/gates/run-gen-q27.log"
if grep -q 'MISMATCH' "$OUT/gates/run-gen-q27.log"; then
    echo "FAIL: Q27 run-gen emitted MISMATCH"
    exit 1
fi

run_logged run-gen-q35 "$OUT/gates/run-gen-q35.log" \
    timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q35"
grep -q 'argmax=.*MATCH' "$OUT/gates/run-gen-q35.log"
if grep -q 'MISMATCH' "$OUT/gates/run-gen-q35.log"; then
    echo "FAIL: Q35 run-gen emitted MISMATCH"
    exit 1
fi

run_logged run-spec-q27 "$OUT/gates/run-spec-q27.log" \
    timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$Q27"
test "$(grep -c 'self-consistency: PASS' "$OUT/gates/run-spec-q27.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/gates/run-spec-q27.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/gates/run-spec-q27.log"; then
    echo "FAIL: Q27 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi

run_logged run-spec-q35 "$OUT/gates/run-spec-q35.log" \
    timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$Q35"
test "$(grep -c 'self-consistency: PASS' "$OUT/gates/run-spec-q35.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/gates/run-spec-q35.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/gates/run-spec-q35.log"; then
    echo "FAIL: Q35 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi
touch "$OUT/gates/gates.ok"

for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        run_model_rep q27 "$rep"
        run_model_rep q35 "$rep"
    else
        run_model_rep q35 "$rep"
        run_model_rep q27 "$rep"
    fi
done

stop_samplers
snapshot "$OUT/gpu-after.log" after
test -z "$(compute_apps_gpu0)" || { compute_apps_gpu0; echo "FAIL: GPU0 process remained"; exit 1; }
touch "$OUT/campaign/campaign.complete"
write_manifest
trap - EXIT INT TERM
echo "REQUAL2_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT"
