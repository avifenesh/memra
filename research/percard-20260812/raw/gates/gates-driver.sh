#!/usr/bin/env bash
# Q27 + Q35 one-per-card gates and simultaneous-serving capacity campaign.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    gates|campaign) ;;
    *) echo "usage: $0 gates|campaign" >&2; exit 2 ;;
esac

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${PERCARD_ROOT:-/opt/dl-image/nvme/cx-percard}
REPO=${PERCARD_REPO:-$ROOT/memra}
MODELS=$ROOT/models
RAW_ROOT=$ROOT/raw
STAMP=${PERCARD_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${PERCARD_OUT:-$RAW_ROOT/$MODE-$STAMP}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BUNDLED_MAIN=250ba819e83f868d395c01c6f315a4c6344f54cb
# v0.78.0 changed only workspace.package.version and omitted the matching path pins.
# Its parent has byte-identical crates/ and tools/ and is the buildable v0.78 runtime tree.
EXPECTED_SOURCE=8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4
EXPECTED_BUNDLE=29a035539c65a167081e0342afb2ad263a4614b1d6c228b517cddce4af4f5d1e

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
PAIR_PROBE=$SCRIPT_DIR/pair_probe.py
GOLDEN_PROBE=$SCRIPT_DIR/golden_probe.py
PORT27=${PERCARD_PORT27:-18270}
PORT35=${PERCARD_PORT35:-18350}
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
    local source
    source=$(git -C "$REPO" rev-parse HEAD)
    echo "source_commit=$source"
    test "$source" = "$EXPECTED_SOURCE"
    git -C "$REPO" diff --quiet
    git -C "$REPO" diff --cached --quiet
    git -C "$REPO" merge-base --is-ancestor "$EXPECTED_SOURCE" "$BUNDLED_MAIN"
    git -C "$REPO" diff --quiet "$EXPECTED_SOURCE" "$BUNDLED_MAIN" -- crates tools
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
    for binary in "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER"; do
        test -x "$binary" || { echo "FAIL: missing binary $binary"; return 1; }
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

write_manifest() {
    local root=$1 temp
    # driver.log is still open through the final PASS line, so hash every stable payload around it.
    temp=$(mktemp "$root/.manifest.XXXXXX")
    (
        cd "$root"
        find . -type f ! -name MANIFEST.sha256 ! -name driver.log ! -name '.manifest.*' \
            -print0 | sort -z | xargs -0 sha256sum
    ) >"$temp"
    mv "$temp" "$root/MANIFEST.sha256"
}

run_gates() {
    echo "PERCARD_GATES_START ts=$(date -u +%FT%TZ) out=$OUT"
    source_preflight
    artifact_preflight | tee "$OUT/artifact-preflight.log"
    binary_preflight
    test "$(sha256sum "$ROOT/memra-main.bundle" | awk '{print $1}')" = "$EXPECTED_BUNDLE"
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        hostname
        uname -a
        echo "bundled_main=$BUNDLED_MAIN"
        echo "scored_runtime_source=$EXPECTED_SOURCE"
        echo "runtime_tools_match_bundled_main=yes"
        git -C "$REPO" log -5 --oneline --decorate
        rustc --version
        cargo --version
        nvcc --version
        df -h "$ROOT"
        stat -c '%s %n' "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader
    } >"$OUT/provenance.txt" 2>&1
    sha256sum "$ROOT/memra-main.bundle" "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" \
        "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" "$PROMPT" \
        "$PAIR_PROBE" "$GOLDEN_PROBE" >"$OUT/SHA256SUMS"

    echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w 1800 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "PERCARD_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    snapshot "$OUT/gpu-before.log" gates-before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPUs not idle"; return 1; }

    run_logged kernel-gpu0 "$OUT/kernel-gpu0.log" \
        timeout 1800 env CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
    grep -q 'ALL GREEN' "$OUT/kernel-gpu0.log"
    if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-gpu0.log"; then
        echo "FAIL: kernel-gpu0 emitted a failure verdict"
        return 1
    fi

    run_logged kernel-gpu1 "$OUT/kernel-gpu1.log" \
        timeout 1800 env CUDA_VISIBLE_DEVICES=1 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL"
    grep -q 'ALL GREEN' "$OUT/kernel-gpu1.log"
    if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-gpu1.log"; then
        echo "FAIL: kernel-gpu1 emitted a failure verdict"
        return 1
    fi

    run_logged run-gen-q27 "$OUT/run-gen-q27.log" \
        timeout 1800 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q27"
    grep -q 'argmax=.*MATCH' "$OUT/run-gen-q27.log"
    if grep -q 'MISMATCH' "$OUT/run-gen-q27.log"; then
        echo "FAIL: Q27 run-gen emitted MISMATCH"
        return 1
    fi

    run_logged run-gen-q35 "$OUT/run-gen-q35.log" \
        timeout 1800 env CUDA_VISIBLE_DEVICES=1 MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_GEN" "$Q35"
    grep -q 'argmax=.*MATCH' "$OUT/run-gen-q35.log"
    if grep -q 'MISMATCH' "$OUT/run-gen-q35.log"; then
        echo "FAIL: Q35 run-gen emitted MISMATCH"
        return 1
    fi

    run_logged run-spec-q27 "$OUT/run-spec-q27.log" \
        timeout 3600 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
            CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$Q27"
    test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q27.log")" -eq 8
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q27.log"
    if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q27.log"; then
        echo "FAIL: Q27 run-spec emitted SELF-CONSISTENCY FAIL"
        return 1
    fi

    run_logged run-spec-q35 "$OUT/run-spec-q35.log" \
        timeout 3600 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
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
    echo "PERCARD_GATES_PASS ts=$(date -u +%FT%TZ) out=$OUT"
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

wait_ready() {
    local base=$1 pid=$2 label=$3 log=$4
    for _ in $(seq 1 360); do
        curl -sf "$base/v1/models" >/dev/null 2>&1 && return 0
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FAIL: $label server died during boot"
            tail -100 "$log"
            return 1
        fi
        sleep 2
    done
    echo "FAIL: $label server never became ready"
    tail -100 "$log"
    return 1
}

start_servers() {
    for port in "$PORT27" "$PORT35"; do
        if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
            echo "FAIL: port $port already has a listener"
            ss -tlnp 2>/dev/null | grep "[:.]$port " || true
            return 1
        fi
    done
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_SERVE_BATCH \
        -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_PREFIX_CACHE_MB -u MEMRA_CTX \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="q27=$Q27+$Q27_DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT27" \
        MEMRA_TAG=cx-percard-q27 "$SERVER" >"$OUT/server-q27.log" 2>&1 &
    server27_pid=$!
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_SERVE_BATCH \
        -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_PREFIX_CACHE_MB -u MEMRA_CTX \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=1 MEMRA_MODELS="q35=$Q35+$Q35_DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT35" \
        MEMRA_TAG=cx-percard-q35 "$SERVER" >"$OUT/server-q35.log" 2>&1 &
    server35_pid=$!
    wait_ready "$BASE27" "$server27_pid" q27 "$OUT/server-q27.log"
    wait_ready "$BASE35" "$server35_pid" q35 "$OUT/server-q35.log"
    ss -tlnp >"$OUT/listeners-after-start.txt" 2>&1
    grep "[:.]$PORT27 " "$OUT/listeners-after-start.txt" | grep -q "pid=$server27_pid,"
    grep "[:.]$PORT35 " "$OUT/listeners-after-start.txt" | grep -q "pid=$server35_pid,"
    compute_apps >"$OUT/compute-apps-after-start.txt"
    curl -sf "$BASE27/v1/models" >"$OUT/models-q27.json"
    curl -sf "$BASE35/v1/models" >"$OUT/models-q35.json"
}

assert_server_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' \
        "$log" || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

probe_endpoints() {
    local condition=$1
    case "$condition" in
        paired)
            printf '%s\n' --endpoint "q27,$BASE27,q27" --endpoint "q35,$BASE35,q35"
            ;;
        q27)
            printf '%s\n' --endpoint "q27,$BASE27,q27"
            ;;
        q35)
            printf '%s\n' --endpoint "q35,$BASE35,q35"
            ;;
        *) return 2 ;;
    esac
}

run_probe() {
    local condition=$1 label=$2 width=$3 output=$4 log=$5
    local args=()
    while IFS= read -r line; do args+=("$line"); done < <(probe_endpoints "$condition")
    run_logged "$label" "$log" timeout 1800 python3 "$PAIR_PROBE" \
        "${args[@]}" --label "$label" --concurrency "$width" \
        --max-tokens 128 --sample-ms 250 --out "$output"
}

run_campaign() {
    test -f "$RAW_ROOT/latest-gates/gates.ok" || {
        echo "FAIL: current gates receipt is absent"; return 1;
    }
    source_preflight
    artifact_preflight >"$OUT/artifact-preflight.log"
    binary_preflight
    echo "PERCARD_CAMPAIGN_START ts=$(date -u +%FT%TZ) out=$OUT"
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        echo "bundled_main=$BUNDLED_MAIN"
        echo "scored_runtime_source=$EXPECTED_SOURCE"
        echo "runtime_tools_match_bundled_main=yes"
        echo "shape=two independent processes; q27 physical GPU0; q35 physical GPU1"
        echo "server_defaults=spec gate on; no PP; finite 128-token requests; unique cache salt"
        git -C "$REPO" log -5 --oneline --decorate
        sha256sum "$SERVER" "$PAIR_PROBE" "$GOLDEN_PROBE"
    } >"$OUT/provenance.txt"
    sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$SERVER" \
        "$PAIR_PROBE" "$GOLDEN_PROBE" >"$OUT/SHA256SUMS"

    echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "PERCARD_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
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

    mkdir -p "$OUT/exactness"
    run_logged golden-q27 "$OUT/exactness/golden-q27.log" timeout 3600 \
        python3 "$GOLDEN_PROBE" --base "$BASE27" --model q27 --label q27 \
            --repeats 10 --max-tokens 128 \
            --out "$OUT/exactness/golden-q27.json" \
            --golden "$OUT/exactness/q27-golden.bin"
    grep -q '"verdict": "PASS"' "$OUT/exactness/golden-q27.json"
    run_logged golden-q35 "$OUT/exactness/golden-q35.log" timeout 3600 \
        python3 "$GOLDEN_PROBE" --base "$BASE35" --model q35 --label q35 \
            --repeats 10 --max-tokens 128 \
            --out "$OUT/exactness/golden-q35.json" \
            --golden "$OUT/exactness/q35-golden.bin"
    grep -q '"verdict": "PASS"' "$OUT/exactness/golden-q35.json"
    curl -sf "$BASE27/metrics" >"$OUT/exactness/metrics-q27-after.json"
    curl -sf "$BASE35/metrics" >"$OUT/exactness/metrics-q35-after.json"
    echo "PERCARD_GOLDENS_PASS ts=$(date -u +%FT%TZ)"

    local width condition position label dir
    local orders=(
        "1 2 4 8 12 16 24"
        "24 16 12 8 4 2 1"
        "8 12 16 24 1 2 4"
    )
    local condition_orders=(
        "paired q27 q35"
        "q27 q35 paired"
        "q35 paired q27"
    )
    mkdir -p "$OUT/perf"
    for rep in 1 2 3; do
        read -r -a widths <<<"${orders[$((rep - 1))]}"
        read -r -a conditions <<<"${condition_orders[$((rep - 1))]}"
        position=0
        for width in "${widths[@]}"; do
            for condition in "${conditions[@]}"; do
                position=$((position + 1))
                label=$(printf 'r%d-p%02d-c%02d-%s' "$rep" "$position" "$width" "$condition")
                dir=$OUT/perf/$label
                mkdir -p "$dir"
                snapshot "$dir/thermal-before.log" "$label-before"
                curl -sf "$BASE27/metrics" >"$dir/metrics-q27-before.json"
                curl -sf "$BASE35/metrics" >"$dir/metrics-q35-before.json"
                run_probe "$condition" "$label-warmup" "$width" \
                    "$dir/warmup.jsonl" "$dir/warmup.log"
                echo "SCORE_START label=$label ts=$(date -u +%FT%TZ)"
                run_probe "$condition" "$label" "$width" \
                    "$dir/score.jsonl" "$dir/score.log"
                curl -sf "$BASE27/metrics" >"$dir/metrics-q27-after.json"
                curl -sf "$BASE35/metrics" >"$dir/metrics-q35-after.json"
                snapshot "$dir/thermal-after.log" "$label-after"
                echo "SCORE_PASS label=$label ts=$(date -u +%FT%TZ)"
            done
        done
    done

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
    echo "PERCARD_CAMPAIGN_PASS ts=$(date -u +%FT%TZ) out=$OUT"
}

case "$MODE" in
    gates) run_gates ;;
    campaign) run_campaign ;;
esac
