#!/usr/bin/env bash
# Q27 + Q35 sold-shape exactness and cache/latency gate on the eu-west PRO pair.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    gates|pilot|campaign) ;;
    *) echo "usage: $0 gates|pilot|campaign" >&2; exit 2 ;;
esac

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${SELLGATE_ROOT:-/opt/dl-image/nvme/cx-sellgate}
REPO=${SELLGATE_REPO:-$ROOT/memra}
HARNESS=${SELLGATE_HARNESS:-$ROOT/harness}
MODELS=${SELLGATE_MODELS:-/opt/dl-image/nvme/cx-percard/models}
RAW_ROOT=$ROOT/raw
STAMP=${SELLGATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${SELLGATE_OUT:-$RAW_ROOT/$MODE-$STAMP}

EXPECTED_SOURCE=79c3c0b2779101c7de89d6f822b9392d03e71702
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
PROMPT_PILOT=$HARNESS/prompt_pilot.py
REPLAY=$HARNESS/sellgate_replay.py
WORKLOAD_LOCK=$HARNESS/workload.lock.json
DRIVER=$HARNESS/run-eu-west.sh
PORT27=${SELLGATE_PORT27:-18427}
PORT35=${SELLGATE_PORT35:-18435}
BASE27=http://127.0.0.1:$PORT27
BASE35=http://127.0.0.1:$PORT35

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

source_preflight() {
    local source dirty
    source=$(git -C "$REPO" rev-parse HEAD)
    echo "source_commit=$source"
    test "$source" = "$EXPECTED_SOURCE"
    dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
    test -z "$dirty" || { echo "$dirty"; echo "FAIL: staged checkout is dirty"; return 1; }
}

check_hash() {
    local expected=$1 path=$2 actual
    test -f "$path"
    actual=$(sha256sum "$path" | awk '{print $1}')
    echo "$actual  $path"
    test "$actual" = "$expected"
}

artifact_preflight() {
    check_hash "$EXPECTED_Q27" "$Q27"
    check_hash "$EXPECTED_Q27_DRAFT" "$Q27_DRAFT"
    check_hash "$EXPECTED_Q35" "$Q35"
    check_hash "$EXPECTED_Q35_DRAFT" "$Q35_DRAFT"
}

binary_preflight() {
    local binary
    for binary in "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER"; do
        test -x "$binary" || { echo "FAIL: missing binary $binary"; return 1; }
    done
    python3 -m py_compile "$PREFIX_GATE" "$PROMPT_PILOT" "$REPLAY"
    python3 -m json.tool "$WORKLOAD_LOCK" >/dev/null
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

write_manifest() {
    local root=$1 temp
    temp=$(mktemp "$root/.manifest.XXXXXX")
    (
        cd "$root"
        find . -type f ! -name MANIFEST.sha256 ! -name driver.log ! -name '.manifest.*' \
            -print0 | sort -z | xargs -0 sha256sum
    ) >"$temp"
    mv "$temp" "$root/MANIFEST.sha256"
}

write_common_provenance() {
    local destination=$1
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        echo "runtime_source=$EXPECTED_SOURCE"
        echo "shape=q27 physical GPU0 plus q35 physical GPU1; two independent servers"
        hostname
        uname -a
        git -C "$REPO" log -5 --oneline --decorate
        rustc --version
        cargo --version
        nvcc --version
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader
        nvidia-smi topo -m
        df -h "$ROOT" "$MODELS"
    } >"$destination" 2>&1
}

run_gates() {
    echo "SELLGATE_GATES_START ts=$(date -u +%FT%TZ) out=$OUT"
    source_preflight
    artifact_preflight | tee "$OUT/artifact-preflight.log"
    binary_preflight
    write_common_provenance "$OUT/provenance.txt"
    sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" \
        "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" "$PROMPT" \
        "$PREFIX_GATE" "$PROMPT_PILOT" "$REPLAY" "$WORKLOAD_LOCK" "$DRIVER" \
        >"$OUT/SHA256SUMS"

    echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "SELLGATE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    snapshot "$OUT/gpu-before.log" gates-before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPUs not idle"; return 1; }

    run_logged kernel-gpu0 "$OUT/kernel-gpu0.log" \
        timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
    grep -q 'ALL GREEN' "$OUT/kernel-gpu0.log"
    if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-gpu0.log"; then
        echo "FAIL: kernel-gpu0 emitted a failure verdict"
        return 1
    fi

    run_logged kernel-gpu1 "$OUT/kernel-gpu1.log" \
        timeout 2400 env CUDA_VISIBLE_DEVICES=1 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
    grep -q 'ALL GREEN' "$OUT/kernel-gpu1.log"
    if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-gpu1.log"; then
        echo "FAIL: kernel-gpu1 emitted a failure verdict"
        return 1
    fi

    run_logged run-gen-q27 "$OUT/run-gen-q27.log" \
        timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q27"
    grep -q 'argmax=.*MATCH' "$OUT/run-gen-q27.log"
    if grep -q 'MISMATCH' "$OUT/run-gen-q27.log"; then
        echo "FAIL: Q27 run-gen emitted MISMATCH"
        return 1
    fi

    run_logged run-gen-q35 "$OUT/run-gen-q35.log" \
        timeout 2400 env CUDA_VISIBLE_DEVICES=1 MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q35"
    grep -q 'argmax=.*MATCH' "$OUT/run-gen-q35.log"
    if grep -q 'MISMATCH' "$OUT/run-gen-q35.log"; then
        echo "FAIL: Q35 run-gen emitted MISMATCH"
        return 1
    fi

    run_logged run-spec-q27 "$OUT/run-spec-q27.log" \
        timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
            CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$Q27"
    test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q27.log")" -eq 8
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q27.log"
    if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q27.log"; then
        echo "FAIL: Q27 run-spec emitted SELF-CONSISTENCY FAIL"
        return 1
    fi

    run_logged run-spec-q35 "$OUT/run-spec-q35.log" \
        timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
            CUDA_VISIBLE_DEVICES=1 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$Q35"
    test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q35.log")" -eq 8
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q35.log"
    if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q35.log"; then
        echo "FAIL: Q35 run-spec emitted SELF-CONSISTENCY FAIL"
        return 1
    fi

    snapshot "$OUT/gpu-after.log" gates-after
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; return 1; }
    touch "$OUT/gates.ok"
    write_manifest "$OUT"
    ln -sfn "$OUT" "$RAW_ROOT/latest-gates"
    echo "SELLGATE_GATES_PASS ts=$(date -u +%FT%TZ) out=$OUT"
}

server27_pid=
server35_pid=
sampler_pid=
vmstat_pid=
dmon_pid=

stop_one_server() {
    local pid=${1:-} label=$2
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server $label pid=$pid did not stop after 120s; sending KILL"
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
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

cleanup_campaign() {
    stop_one_server "${server27_pid:-}" q27 || true
    stop_one_server "${server35_pid:-}" q35 || true
    stop_samplers
}

run_prompt_pilot() {
    source_preflight
    artifact_preflight >"$OUT/artifact-preflight.log"
    binary_preflight
    echo "SELLGATE_PROMPT_PILOT_START ts=$(date -u +%FT%TZ) out=$OUT"
    write_common_provenance "$OUT/provenance.txt"
    sha256sum "$Q27" "$Q35" "$SERVER" "$PROMPT_PILOT" "$REPLAY" \
        "$WORKLOAD_LOCK" "$DRIVER" >"$OUT/SHA256SUMS.input"

    echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "SELLGATE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    snapshot "$OUT/gpu-before.log" prompt-pilot-before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPUs not idle"; return 1; }

    trap cleanup_campaign EXIT INT TERM
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
    sampler_pid=$!
    start_servers
    snapshot "$OUT/gpu-servers-ready.log" prompt-pilot-servers-ready
    run_logged prompt-pilot "$OUT/prompt-pilot.log" timeout 14400 python3 "$PROMPT_PILOT" \
        --endpoint "q27,$BASE27,q27" --endpoint "q35,$BASE35,q35" \
        --workload-lock "$WORKLOAD_LOCK" --out "$OUT/prompt-pilot.jsonl" \
        --namespace cx-sellgate-prompt-pilot --offset 105 --repetitions 3 \
        --concurrency 1,2,4,8 --timeout 1800
    grep -q '"verdict": "PASS"' "$OUT/prompt-pilot.jsonl"

    stop_one_server "$server27_pid" q27
    server27_pid=
    stop_one_server "$server35_pid" q35
    server35_pid=
    stop_samplers
    assert_server_clean "$OUT/server-q27.log"
    assert_server_clean "$OUT/server-q35.log"
    snapshot "$OUT/gpu-after.log" prompt-pilot-after
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; return 1; }
    touch "$OUT/pilot.ok"
    write_manifest "$OUT"
    ln -sfn "$OUT" "$RAW_ROOT/latest-pilot"
    trap - EXIT INT TERM
    echo "SELLGATE_PROMPT_PILOT_PASS ts=$(date -u +%FT%TZ) out=$OUT"
}

wait_ready() {
    local base=$1 pid=$2 label=$3 log=$4
    for _ in $(seq 1 900); do
        curl -sf "$base/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FAIL: $label server died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: $label server never became ready"
    tail -200 "$log"
    return 1
}

start_servers() {
    local port
    for port in "$PORT27" "$PORT35"; do
        if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
            echo "FAIL: port $port already has a listener"
            ss -tlnp 2>/dev/null | grep "[:.]$port " || true
            return 1
        fi
    done
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST -u MEMRA_MOE_RESIDENT \
        -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="q27=$Q27" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT27" \
        MEMRA_TAG=cx-sellgate-q27 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$OUT/server-q27.log" 2>&1 &
    server27_pid=$!
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST -u MEMRA_MOE_RESIDENT \
        -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=1 MEMRA_MODELS="q35=$Q35" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT35" \
        MEMRA_TAG=cx-sellgate-q35 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$OUT/server-q35.log" 2>&1 &
    server35_pid=$!
    wait_ready "$BASE27" "$server27_pid" q27 "$OUT/server-q27.log"
    wait_ready "$BASE35" "$server35_pid" q35 "$OUT/server-q35.log"
    ss -tlnp >"$OUT/listeners-after-start.txt" 2>&1
    grep "[:.]$PORT27 " "$OUT/listeners-after-start.txt" | grep -q "pid=$server27_pid,"
    grep "[:.]$PORT35 " "$OUT/listeners-after-start.txt" | grep -q "pid=$server35_pid,"
    compute_apps >"$OUT/compute-apps-after-start.txt"
    curl -sf "$BASE27/v1/models" >"$OUT/models-q27.json"
    curl -sf "$BASE35/v1/models" >"$OUT/models-q35.json"
    grep -q '\[prefix-cache\] on:' "$OUT/server-q27.log"
    grep -q '\[prefix-cache\] on:' "$OUT/server-q35.log"
}

assert_server_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' \
        "$log" || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

run_prefix_exactness() {
    mkdir -p "$OUT/exactness"
    echo "PREFIX_EXACTNESS_START ts=$(date -u +%FT%TZ)"
    set +e
    timeout 7200 python3 "$PREFIX_GATE" \
        --endpoint "q27,$BASE27,q27" --workload-lock "$WORKLOAD_LOCK" \
        --out "$OUT/exactness/q27.jsonl" --namespace cx-sellgate-q27-exact \
        >"$OUT/exactness/q27.log" 2>&1 &
    local exact27_pid=$!
    timeout 7200 python3 "$PREFIX_GATE" \
        --endpoint "q35,$BASE35,q35" --workload-lock "$WORKLOAD_LOCK" \
        --out "$OUT/exactness/q35.jsonl" --namespace cx-sellgate-q35-exact \
        >"$OUT/exactness/q35.log" 2>&1 &
    local exact35_pid=$!
    wait "$exact27_pid"
    local rc27=$?
    wait "$exact35_pid"
    local rc35=$?
    set -e
    echo "$rc27" >"$OUT/exactness/q27.exit"
    echo "$rc35" >"$OUT/exactness/q35.exit"
    local passes=0
    if test "$rc27" -eq 0 && grep -q '"verdict": "PASS"' "$OUT/exactness/q27.jsonl"; then
        passes=$((passes + 1))
    fi
    if test "$rc35" -eq 0 && grep -q '"verdict": "PASS"' "$OUT/exactness/q35.jsonl"; then
        passes=$((passes + 1))
    fi
    test "$passes" -ge 1 || {
        echo "FAIL: neither model passed serial prefix-cache exactness"
        return 1
    }
    echo "$passes" >"$OUT/exactness/models-passed"
    echo "PREFIX_EXACTNESS_PASS models=$passes ts=$(date -u +%FT%TZ)"
}

run_campaign() {
    test -f "$RAW_ROOT/latest-gates/gates.ok" || {
        echo "FAIL: current sellgate gates receipt is absent"; return 1;
    }
    source_preflight
    artifact_preflight >"$OUT/artifact-preflight.log"
    binary_preflight
    echo "SELLGATE_CAMPAIGN_START ts=$(date -u +%FT%TZ) out=$OUT"
    write_common_provenance "$OUT/provenance.txt"
    {
        echo "serve_shape=one target-only plain server per physical GPU; both active"
        echo "draft_shape=external drafters are exactness-gate inputs, not loaded by cache serving"
        echo "MEMRA_SERVE_SPEC=0 (cross-request prefix cache excludes spec sessions)"
        echo "MEMRA_PREFIX_CACHE_MB=4096 per model"
        echo "MEMRA_PREFIX_DEDUP=1"
        echo "MEMRA_REUSE_POOL=0"
        echo "MEMRA_AFFINITY=0"
        echo "MEMRA_CTX=8192"
        echo "MEMRA_MAX_SESSIONS=96"
        echo "workload_lock_sha256=$(sha256sum "$WORKLOAD_LOCK" | awk '{print $1}')"
        echo "exactness_prompt_sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
    } >>"$OUT/provenance.txt"
    sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$SERVER" \
        "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$PROMPT" "$PREFIX_GATE" \
        "$PROMPT_PILOT" "$REPLAY" "$WORKLOAD_LOCK" "$DRIVER" \
        >"$OUT/SHA256SUMS.input"

    echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "SELLGATE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    snapshot "$OUT/gpu-before.log" campaign-before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPUs not idle"; return 1; }

    trap cleanup_campaign EXIT INT TERM
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
    sampler_pid=$!
    vmstat 1 >"$OUT/vmstat-1s.log" 2>&1 &
    vmstat_pid=$!
    nvidia-smi dmon -s pucm -d 1 -o DT >"$OUT/pcie-dmon-1s.log" 2>&1 &
    dmon_pid=$!

    start_servers
    snapshot "$OUT/gpu-servers-ready.log" servers-ready
    curl -sf "$BASE27/metrics" >"$OUT/metrics-q27-before.json"
    curl -sf "$BASE35/metrics" >"$OUT/metrics-q35-before.json"
    run_prefix_exactness
    curl -sf "$BASE27/metrics" >"$OUT/metrics-q27-after-exactness.json"
    curl -sf "$BASE35/metrics" >"$OUT/metrics-q35-after-exactness.json"

    run_logged replay "$OUT/replay.log" timeout 86400 python3 "$REPLAY" \
        --endpoint "q27,$BASE27,q27" --endpoint "q35,$BASE35,q35" \
        --workload-lock "$WORKLOAD_LOCK" --out "$OUT/replay.jsonl" \
        --namespace cx-sellgate-scored --timeout 1800
    grep -q '"verdict": "PASS"' "$OUT/replay.jsonl"
    curl -sf "$BASE27/metrics" >"$OUT/metrics-q27-final.json"
    curl -sf "$BASE35/metrics" >"$OUT/metrics-q35-final.json"

    stop_one_server "$server27_pid" q27
    server27_pid=
    stop_one_server "$server35_pid" q35
    server35_pid=
    stop_samplers
    assert_server_clean "$OUT/server-q27.log"
    assert_server_clean "$OUT/server-q35.log"
    snapshot "$OUT/gpu-after.log" campaign-after
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; return 1; }
    touch "$OUT/campaign.ok"
    write_manifest "$OUT"
    ln -sfn "$OUT" "$RAW_ROOT/latest-campaign"
    trap - EXIT INT TERM
    echo "SELLGATE_CAMPAIGN_PASS ts=$(date -u +%FT%TZ) out=$OUT"
}

case "$MODE" in
    gates) run_gates ;;
    pilot) run_prompt_pilot ;;
    campaign) run_campaign ;;
esac
