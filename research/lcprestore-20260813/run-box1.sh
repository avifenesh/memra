#!/usr/bin/env bash
# Complete lcprestore campaign on physical box1 GPU 1. The caller stages the committed source.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${LCPRESTORE_ROOT:-/opt/scratch/nvme/cx-lcprestore}
REPO=${LCPRESTORE_REPO:-$ROOT/memra}
MODELS=${LCPRESTORE_MODELS:-/opt/scratch/nvme/cx-requal/models}
EXPECTED_SOURCE=${LCPRESTORE_EXPECTED_SOURCE:?set LCPRESTORE_EXPECTED_SOURCE}
STAMP=${LCPRESTORE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${LCPRESTORE_OUT:-$ROOT/raw/run-$STAMP}

GPU_PHYSICAL=1
GPU_UUID=GPU-2b4cf166-fd33-f161-8536-ca04bc72280c
GPU_LOCK=/tmp/memra-gpu-1.lock

TRANSFORMER=$ROOT/models/gemma-4-12b-it-qat-q4_0.gguf
TRANSFORMER_SHA256=93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b
Q27=$MODELS/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=$MODELS/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=$MODELS/draft-35b-owntrim-nvfp4head-q4blk.gguf
Q27_SHA256=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
Q27_DRAFT_SHA256=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
Q35_SHA256=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
Q35_DRAFT_SHA256=ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a

SERVER=$REPO/target/release/memra-server
KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
PROMPT=$REPO/research/e2e/prompts/pp512.txt
LANE=$REPO/research/lcprestore-20260813
WORKLOAD=$REPO/research/sellgate-20260812/workload.lock.json
PREFIX_GATE=$REPO/research/sellgate-20260812/prefix_exactness.py
FROZEN_REPLAY=$REPO/research/sellgate-20260812/sellgate_replay.py
SINGLE_REPLAY=$REPO/research/requal2-20260812/single_replay.py

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"/{build,exactness,performance,hybrid,mixed/control,mixed/candidate,gates,gpu}
exec > >(tee "$OUT/orchestrator.log") 2>&1

server_pids=()
sampler_pid=
vmstat_pid=
dmon_pid=

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ)"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "RUN_DONE label=$label rc=$rc ts=$(date -u +%FT%TZ)"
    return "$rc"
}

check_hash() {
    local expected=$1 path=$2 actual
    test -f "$path"
    actual=$(sha256sum "$path" | awk '{print $1}')
    echo "$actual  $path"
    test "$actual" = "$expected"
}

compute_apps() {
    nvidia-smi -i "$GPU_PHYSICAL" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

assert_no_foreign() {
    local label=$1
    shift
    local apps allowed pid line ok
    apps=$(compute_apps)
    {
        echo "label=$label ts=$(date -u +%FT%TZ) physical_gpu=$GPU_PHYSICAL uuid=$GPU_UUID"
        if test -n "$apps"; then printf '%s\n' "$apps"; else echo "compute_apps=none"; fi
    } | tee -a "$OUT/gpu/compute-app-preflights.log"
    while IFS= read -r line; do
        test -n "$line" || continue
        pid=${line%%,*}
        pid=${pid// /}
        ok=0
        for allowed in "$@"; do
            if test "$pid" = "$allowed"; then ok=1; break; fi
        done
        if test "$ok" -ne 1; then
            echo "FAIL: foreign compute process before $label: $line"
            return 1
        fi
    done <<<"$apps"
}

snapshot_gpu() {
    local label=$1 path=$2
    {
        echo "label=$label ts=$(date -u +%FT%TZ)"
        nvidia-smi -i "$GPU_PHYSICAL" \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } | tee "$path"
}

wait_ready() {
    local pid=$1 base=$2 log=$3
    for _ in $(seq 1 900); do
        curl -sf "$base/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FAIL: server pid=$pid died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server pid=$pid never became ready"
    tail -200 "$log"
    return 1
}

start_server() {
    local name=$1 model=$2 port=$3 partial=$4 trace=$5 log=$6
    if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
        echo "FAIL: port $port already has a listener"
        return 1
    fi
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" MEMRA_MODELS="$name=$model" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$port" MEMRA_CTX=8192 \
        MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=8192 MEMRA_PREFIX_DEDUP=1 \
        MEMRA_PREFIX_PARTIAL_RESTORE="$partial" MEMRA_PREFIX_SPLIT_TRACE="$trace" \
        MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" > >(tee "$log" >/dev/null) 2>&1 &
    STARTED_PID=$!
    server_pids+=("$STARTED_PID")
    wait_ready "$STARTED_PID" "http://127.0.0.1:$port" "$log"
}

stop_server() {
    local pid=$1
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server pid=$pid did not stop after 120s; sending KILL"
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 1
}

stop_all_servers() {
    local pid
    for pid in "${server_pids[@]:-}"; do
        test -n "$pid" || continue
        stop_server "$pid" || true
    done
    server_pids=()
    for _ in $(seq 1 60); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
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

assert_server_log_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' \
        "$log" || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

assert_ports_clear() {
    local port
    for port in 8177 8179 18731 18732 18735 18741; do
        if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
            echo "FAIL: lane port $port remains open"
            return 1
        fi
    done
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_all_servers || true
    stop_samplers
    snapshot_gpu cleanup "$OUT/gpu/cleanup.log" || true
    assert_ports_clear || true
    flock -u 9 2>/dev/null || true
    exec 9>&- || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "LCPRESTORE_PREFLIGHT ts=$(date -u +%FT%TZ) host=$(hostname) source=$EXPECTED_SOURCE"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
test "$(nvidia-smi -i "$GPU_PHYSICAL" --query-gpu=uuid --format=csv,noheader | tr -d ' ')" = "$GPU_UUID"
check_hash "$TRANSFORMER_SHA256" "$TRANSFORMER"
check_hash "$Q27_SHA256" "$Q27"
check_hash "$Q27_DRAFT_SHA256" "$Q27_DRAFT"
check_hash "$Q35_SHA256" "$Q35"
check_hash "$Q35_DRAFT_SHA256" "$Q35_DRAFT"
python3 -m py_compile "$LANE"/*.py "$PREFIX_GATE" "$FROZEN_REPLAY" "$SINGLE_REPLAY"
python3 -m json.tool "$WORKLOAD" >/dev/null
cp "$ROOT/source/gemma-4-12B-it-qat-q4_0-gguf-api.json" \
    "$OUT/build/gemma-4-12B-it-qat-q4_0-gguf-api.json"
cp "$ROOT/source/gemma-4-12b-config.txt" "$OUT/build/gemma-4-12b-config.txt"
sha256sum "$TRANSFORMER" "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" \
    | tee "$OUT/build/model-sha256.txt"
git -C "$REPO" status --short --branch | tee "$OUT/build/git-status.txt"
git -C "$REPO" show --no-patch --format=fuller HEAD | tee "$OUT/build/source-commit.txt"

run_logged cargo-test "$OUT/build/cargo-test.log" \
    env CUDA_VISIBLE_DEVICES= cargo test --manifest-path "$REPO/Cargo.toml" --workspace
run_logged release-server "$OUT/build/release-server.log" \
    cargo build --manifest-path "$REPO/Cargo.toml" --release -p memra-server
run_logged release-gates "$OUT/build/release-gates.log" \
    cargo build --manifest-path "$REPO/Cargo.toml" --release -p memra-engine \
        --bin kernel-check --bin run-gen --bin run-spec
for binary in "$SERVER" "$KERNEL" "$RUN_GEN" "$RUN_SPEC"; do test -x "$binary"; done

exec 9>"$GPU_LOCK"
flock 9
echo "LCPRESTORE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) lock=$GPU_LOCK physical_gpu=$GPU_PHYSICAL uuid=$GPU_UUID"
assert_no_foreign lock-acquired
snapshot_gpu before "$OUT/gpu/before.log"

nvidia-smi -i "$GPU_PHYSICAL" \
    --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu/250ms.csv" 2>&1 &
sampler_pid=$!
vmstat 1 >"$OUT/gpu/vmstat-1s.log" 2>&1 &
vmstat_pid=$!
nvidia-smi dmon -i "$GPU_PHYSICAL" -s pucm -d 1 -o DT >"$OUT/gpu/dmon-1s.log" 2>&1 &
dmon_pid=$!

# HIRADIX-EXACT-ISO at each split. State tracing D2H-hashes every K/V byte, so it is kept
# out of the separate latency cell below.
start_server g12 "$TRANSFORMER" 18731 0 1 "$OUT/exactness/control-server.log"
control_pid=$STARTED_PID
start_server g12 "$TRANSFORMER" 18732 1 1 "$OUT/exactness/candidate-server.log"
candidate_pid=$STARTED_PID
assert_no_foreign split-exactness "$control_pid" "$candidate_pid"
snapshot_gpu split-exactness-before "$OUT/gpu/split-exactness-before.log"
run_logged split-exactness "$OUT/exactness/requests.log" \
    timeout 14400 python3 "$LANE/split_exactness.py" \
        --control g12-control,http://127.0.0.1:18731,g12 \
        --candidate g12-candidate,http://127.0.0.1:18732,g12 \
        --workload-lock "$WORKLOAD" --out "$OUT/exactness/requests.jsonl" \
        --namespace cx-lcprestore-exact --splits 64,512,2048,4374 \
        --main-split 4374 --repetitions 1 --timeout 1800 \
        --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID"
stop_all_servers
assert_server_log_clean "$OUT/exactness/control-server.log"
assert_server_log_clean "$OUT/exactness/candidate-server.log"
run_logged split-state-verify "$OUT/exactness/split-state-verify.log" \
    python3 "$LANE/verify_split_receipts.py" \
        --control-log "$OUT/exactness/control-server.log" \
        --candidate-log "$OUT/exactness/candidate-server.log" \
        --requests "$OUT/exactness/requests.jsonl" \
        --out "$OUT/exactness/split-state-receipts.json" \
        --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID" \
        --gpu-lock "$GPU_LOCK"
grep -q '"verdict": "PASS"' "$OUT/exactness/split-state-receipts.json"

# N=5 sequential request-2 target shape, trace-free. Both servers stay resident and each cell
# flips arm order; feature flag is the only runtime difference.
start_server g12 "$TRANSFORMER" 18731 0 0 "$OUT/performance/control-server.log"
control_pid=$STARTED_PID
start_server g12 "$TRANSFORMER" 18732 1 0 "$OUT/performance/candidate-server.log"
candidate_pid=$STARTED_PID
assert_no_foreign sequential-request2 "$control_pid" "$candidate_pid"
snapshot_gpu sequential-request2-before "$OUT/gpu/sequential-request2-before.log"
run_logged sequential-request2 "$OUT/performance/requests.log" \
    timeout 14400 python3 "$LANE/split_exactness.py" \
        --control g12-control,http://127.0.0.1:18731,g12 \
        --candidate g12-candidate,http://127.0.0.1:18732,g12 \
        --workload-lock "$WORKLOAD" --out "$OUT/performance/requests.jsonl" \
        --namespace cx-lcprestore-performance --splits 4374 \
        --main-split 4374 --repetitions 5 --require-timing-n --timeout 1800 \
        --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID"
stop_all_servers
assert_server_log_clean "$OUT/performance/control-server.log"
assert_server_log_clean "$OUT/performance/candidate-server.log"
grep -q '"verdict": "PASS"' "$OUT/performance/requests.jsonl"

# Hybrid and routed-MoE are explicit, live negative controls.
for target in q27 q35; do
    if test "$target" = q27; then model=$Q27; port=18741; refusal='hybrid conv/SSM'; else model=$Q35; port=18735; refusal='routed-MoE'; fi
    log="$OUT/hybrid/$target-server.log"
    start_server "$target" "$model" "$port" 1 0 "$log"
    pid=$STARTED_PID
    assert_no_foreign "$target-refusal" "$pid"
    run_logged "$target-refusal" "$OUT/hybrid/$target-gate.log" \
        timeout 7200 python3 "$PREFIX_GATE" \
            --endpoint "$target,http://127.0.0.1:$port,$target" \
            --workload-lock "$WORKLOAD" --out "$OUT/hybrid/$target.raw.jsonl" \
            --namespace "cx-lcprestore-$target-refusal" --timeout 1800
    stop_all_servers
    assert_server_log_clean "$log"
    grep -q '"verdict": "PASS"' "$OUT/hybrid/$target.raw.jsonl"
    grep -q "partial restore REFUSED.*$refusal" "$log"
    python3 "$LANE/annotate_gpu.py" --input "$OUT/hybrid/$target.raw.jsonl" \
        --out "$OUT/hybrid/$target.jsonl" --physical-gpu "$GPU_PHYSICAL" \
        --gpu-uuid "$GPU_UUID" --lock "$GPU_LOCK"
done

# Frozen sold-shape Q27 replay, five feature-off/on repetitions. One model resident at a time;
# arm order flips each repetition, all under this same uninterrupted GPU-1 lock.
for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then orders=(control candidate); else orders=(candidate control); fi
    for feature_arm in "${orders[@]}"; do
        if test "$feature_arm" = control; then partial=0; else partial=1; fi
        arm_dir="$OUT/mixed/$feature_arm"
        server_log="$arm_dir/r$(printf '%02d' "$rep")-server.log"
        start_server q27 "$Q27" 18741 "$partial" 0 "$server_log"
        pid=$STARTED_PID
        assert_no_foreign "mixed-$feature_arm-r$rep" "$pid"
        snapshot_gpu "mixed-$feature_arm-r$rep-before" \
            "$OUT/gpu/mixed-$feature_arm-r$(printf '%02d' "$rep")-before.log"
        run_logged "mixed-$feature_arm-r$rep" "$arm_dir/r$(printf '%02d' "$rep").log" \
            timeout 21600 python3 "$SINGLE_REPLAY" \
                --endpoint q27,http://127.0.0.1:18741,q27 \
                --frozen-replay "$FROZEN_REPLAY" --workload-lock "$WORKLOAD" \
                --out "$arm_dir/r$(printf '%02d' "$rep").raw.jsonl" \
                --namespace "cx-lcprestore-$feature_arm-r$rep" --target q27 \
                --levels 1,2,4,8,12,16,20 --repetition "$rep" --timeout 1800
        python3 "$LANE/annotate_gpu.py" \
            --input "$arm_dir/r$(printf '%02d' "$rep").raw.jsonl" \
            --out "$arm_dir/r$(printf '%02d' "$rep").jsonl" \
            --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID" --lock "$GPU_LOCK"
        stop_all_servers
        assert_server_log_clean "$server_log"
    done
done
run_logged mixed-reduce "$OUT/mixed/reduce.log" \
    python3 "$LANE/reduce_mixed.py" --control "$OUT/mixed/control" \
        --candidate "$OUT/mixed/candidate" --frozen-replay "$FROZEN_REPLAY" \
        --out "$OUT/mixed/summary.json" --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID"
grep -q '"verdict": "PASS"' "$OUT/mixed/summary.json"

# Standard correctness and serving battery on the assigned card.
assert_no_foreign kernel-check
run_logged kernel-check "$OUT/gates/kernel-check.log" \
    timeout 2400 env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
grep -q 'ALL GREEN' "$OUT/gates/kernel-check.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/gates/kernel-check.log"; then
    echo "FAIL: kernel-check emitted a failure marker"
    exit 1
fi

for target in q27 q35; do
    if test "$target" = q27; then model=$Q27; draft=$Q27_DRAFT; else model=$Q35; draft=$Q35_DRAFT; fi
    assert_no_foreign "run-gen-$target"
    run_logged "run-gen-$target" "$OUT/gates/run-gen-$target.log" \
        timeout 2400 env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$model"
    grep -q 'argmax=.*MATCH' "$OUT/gates/run-gen-$target.log"
    if grep -q 'MISMATCH' "$OUT/gates/run-gen-$target.log"; then
        echo "FAIL: run-gen-$target emitted MISMATCH"
        exit 1
    fi

    assert_no_foreign "run-spec-$target"
    run_logged "run-spec-$target" "$OUT/gates/run-spec-$target.log" \
        timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
            CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" MEMRA_MTP_DRAFT="$draft" MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$model"
    test "$(grep -c 'self-consistency: PASS' "$OUT/gates/run-spec-$target.log")" -eq 8
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/gates/run-spec-$target.log"
    if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/gates/run-spec-$target.log"; then
        echo "FAIL: run-spec-$target emitted SELF-CONSISTENCY FAIL"
        exit 1
    fi
done

assert_no_foreign serve-smoke
assert_ports_clear
run_logged serve-smoke "$OUT/gates/serve-smoke.log" \
    timeout 21600 env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
        MEMRA_Q35_COLD_MODEL="$Q35" MEMRA_PREFIX_PARTIAL_RESTORE=1 \
        bash "$REPO/tools/serve-smoke.sh" "$Q27" "$Q27_DRAFT"
grep -q 'serve-smoke: 0 failed' "$OUT/gates/serve-smoke.log"

assert_no_foreign c64-stress
run_logged c64-stress "$OUT/gates/c64-stress.log" \
    timeout 21600 env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
        MEMRA_PREFIX_PARTIAL_RESTORE=1 \
        MEMRA_STRESS_LOG="$OUT/gates/c64-server.log" \
        MEMRA_STRESS_ROWS="$OUT/gates/c64.raw.jsonl" \
        bash "$REPO/tools/serve-stress-gate.sh" "$Q27" "$Q27_DRAFT" 64
grep -q 'serve-stress-gate: ALL GREEN' "$OUT/gates/c64-stress.log"
python3 "$LANE/annotate_gpu.py" --input "$OUT/gates/c64.raw.jsonl" \
    --out "$OUT/gates/c64.jsonl" --physical-gpu "$GPU_PHYSICAL" \
    --gpu-uuid "$GPU_UUID" --lock "$GPU_LOCK"

stop_samplers
snapshot_gpu after "$OUT/gpu/after.log"
assert_no_foreign final
test "$(nvidia-smi -i "$GPU_PHYSICAL" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0
assert_ports_clear

(
    cd "$OUT"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) >"$OUT/SHA256SUMS"
touch "$OUT/CAMPAIGN.COMPLETE"
flock -u 9
exec 9>&-
trap - EXIT INT TERM
echo "LCPRESTORE_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT physical_gpu=$GPU_PHYSICAL uuid=$GPU_UUID"
