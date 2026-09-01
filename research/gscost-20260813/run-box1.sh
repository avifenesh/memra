#!/usr/bin/env bash
# One-lock box1 campaign: cold decode A/B, mixed-cache A/B, then the release battery.
set -Eeu -o pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the source commit used for the build}"
ROOT=${GSCOST_ROOT:-/opt/scratch/nvme/cx-gscost}
REPO=${GSCOST_REPO:-$ROOT/memra}
OUT=${GSCOST_OUT:-$REPO/research/gscost-20260813/raw/box1}
MODEL_ROOT=${GSCOST_MODEL_ROOT:-/opt/scratch/nvme/cx-requal/models}
Q27=$MODEL_ROOT/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q35=$MODEL_ROOT/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q27_DRAFT=$MODEL_ROOT/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35_DRAFT=$MODEL_ROOT/draft-35b-owntrim-nvfp4head-q4blk.gguf
SERVER=$REPO/target/release/memra-server
KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
BENCH=$REPO/research/newboxgates-20260811/serve_bench.py
CACHE_HIT_BENCH=$REPO/research/gscost-20260813/cache-hit-decode.py
CACHE_BENCH=$REPO/research/gscost-20260813/cache-single.py
REDUCE=$REPO/research/gscost-20260813/reduce.py
PROMPT=$REPO/research/e2e/prompts/pp512.txt
SELLGATE_MODULE=/opt/scratch/nvme/cx-requal/harness/sellgate_replay.py
WORKLOAD_LOCK=/opt/scratch/nvme/cx-requal/harness/workload.lock.json
PORT=${GSCOST_PORT:-18468}
BASE=http://127.0.0.1:$PORT

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH

for artifact in "$Q27" "$Q35" "$Q27_DRAFT" "$Q35_DRAFT" "$SERVER" "$KERNEL" \
    "$RUN_GEN" "$RUN_SPEC" "$BENCH" "$CACHE_HIT_BENCH" "$CACHE_BENCH" "$REDUCE" "$PROMPT" \
    "$SELLGATE_MODULE" "$WORKLOAD_LOCK"; do
    test -f "$artifact" || { echo "FAIL: missing artifact $artifact"; exit 1; }
done
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
test ! -e "$OUT" || { echo "FAIL: campaign output already exists: $OUT"; exit 1; }

mkdir -p "$OUT/main" "$OUT/mixed" "$OUT/activation" "$OUT/qualification" "$OUT/battery"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=
sampler_pid=
lock_acquired=0
GPU0_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d '[:space:]')

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
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
        ss -ltnp | grep -E ":${PORT}\\b" || true
    } >"$path" 2>&1
}

wait_idle() {
    local apps
    for _ in $(seq 1 180); do
        apps=$(compute_apps)
        test -z "$apps" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

assert_idle() {
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU compute applications present"; return 1; }
}

assert_port_clear() {
    if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
        ss -ltnp 2>/dev/null | grep -E ":${PORT}[[:space:]]" || true
        echo "FAIL: port $PORT is occupied"
        return 1
    fi
}

assert_owned_server() {
    local apps pid uuid bad=0 count=0
    apps=$(compute_apps)
    test -n "$apps" || { echo "FAIL: server has no visible CUDA context"; return 1; }
    while IFS=, read -r pid uuid _; do
        pid=$(echo "$pid" | xargs)
        uuid=$(echo "$uuid" | xargs)
        count=$((count + 1))
        if [[ $pid != "$server_pid" || $uuid != "$GPU0_UUID" ]]; then
            bad=1
        fi
    done <<<"$apps"
    if (( bad != 0 || count != 1 )); then
        echo "$apps"
        echo "FAIL: compute-app census is not exactly the owned GPU0 server pid=$server_pid"
        return 1
    fi
}

stop_sampler() {
    local pid=${sampler_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sampler_pid=
}

stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server pid=$pid did not stop after 120 seconds"
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    server_pid=
    wait_idle || true
    return 1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_server
    stop_sampler
    snapshot "$OUT/cleanup-trap.log" "trap-rc-$rc"
    if (( lock_acquired == 1 )); then
        flock -u 9
        exec 9>&-
    fi
    echo "CAMPAIGN_EXIT rc=$rc ts=$(date -u +%FT%TZ)"
    exit "$rc"
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server did not become ready"
    tail -200 "$log"
    return 1
}

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|sentinel' \
        "$log" || true)
    if [[ -n $failures ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_logged() {
    local log=$1
    shift
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$log.exit"
    return "$rc"
}

start_server() {
    local model=$1 arm=$2 cache_mb=$3 label=$4 census=${5:-0}
    local model_path log env_path
    case "$model" in
        q27) model_path=$Q27 ;;
        q35) model_path=$Q35 ;;
        *) echo "FAIL: unknown model $model"; return 1 ;;
    esac
    log="$OUT/$label-server.log"
    env_path="$OUT/$label-env.txt"
    local -a policy=()
    if [[ $arm == eager ]]; then
        policy=(MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1)
    elif [[ $arm != repaired ]]; then
        echo "FAIL: unknown policy arm $arm"
        return 1
    fi
    if (( census == 1 )); then
        policy+=(MEMRA_GRAPH_CENSUS=1)
    fi
    assert_idle
    assert_port_clear
    {
        echo "label=$label"
        echo "model=$model"
        echo "model_path=$model_path"
        echo "policy_arm=$arm"
        if [[ $arm == eager ]]; then
            echo "MEMRA_SERVE_B1FAST=1"
            echo "MEMRA_SERVE_GS=1"
        else
            echo "MEMRA_SERVE_B1FAST=<unset>"
            echo "MEMRA_SERVE_GS=<unset>"
        fi
        echo "MEMRA_GS_MIN=<unset; default 384>"
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_PREFIX_CACHE_MB=$cache_mb"
        echo "MEMRA_PREFIX_DEDUP=1"
        echo "MEMRA_REUSE_POOL=0"
        echo "MEMRA_AFFINITY=0"
        echo "MEMRA_CTX=8192"
        echo "MEMRA_MAX_SESSIONS=96"
        echo "MEMRA_GRAPH_CENSUS=$([[ $census == 1 ]] && echo 1 || echo '<unset>')"
    } >"$env_path"

    env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS -u MEMRA_GS_MIN \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST -u MEMRA_MOE_GROUPED \
        -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB -u MEMRA_PP_STAGES \
        -u MEMRA_PP_DEVICES -u MEMRA_PP_SPLITS -u MEMRA_PP_SPLIT \
        -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_PRIME_PIPE -u MEMRA_BATCH_PP -u MEMRA_SPEC_PP -u MEMRA_EVT \
        -u MEMRA_GRAPH_CENSUS "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="$model=$model_path" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_CTX=8192 MEMRA_PREFIX_CACHE_MB="$cache_mb" MEMRA_PREFIX_DEDUP=1 \
        MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    echo "$server_pid" >"$OUT/$label-server.pid"
    wait_ready "$log"
    assert_owned_server
    if (( cache_mb > 0 )); then
        grep -q '\[prefix-cache\] on:' "$log"
    fi
}

run_warmup() {
    local label=$1 model=$2
    run_logged "$OUT/$label-warmup.log" python3 "$BENCH" \
        --base "$BASE" --model "$model" --shape warmup --label "$label-warmup" \
        --out "$OUT/warmups.jsonl" --concurrency 1 --max-tokens 16 --timeout 1800
}

run_activation_probe() {
    local model=$1
    local label="activation-$model"
    echo "ACTIVATION_START model=$model ts=$(date -u +%FT%TZ)"
    start_server "$model" eager 4096 "activation/$label" 1
    run_logged "$OUT/activation/$label-load.log" python3 "$CACHE_HIT_BENCH" \
        --base "$BASE" --model "$model" --target "$model" --policy-arm eager \
        --rep 0 --concurrency 1 --max-tokens 512 --label "$label" \
        --namespace "gscost-$label" --out "$OUT/activation-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/activation/$label-metrics.json"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/activation/$label-server.log"
    grep -q '\[graph-census\]' "$OUT/activation/$label-server.log" || {
        echo "FAIL: $model EAGER activation probe did not capture a GraphSession"
        return 1
    }
    echo "ACTIVATION_PASS model=$model ts=$(date -u +%FT%TZ)"
}

run_length_qualification() {
    local model=$1 concurrency=$2
    local label="qualification-$model-c$concurrency"
    echo "QUALIFICATION_START model=$model concurrency=$concurrency ts=$(date -u +%FT%TZ)"
    start_server "$model" eager 4096 "qualification/$label"
    run_logged "$OUT/qualification/$label-load.log" python3 "$CACHE_HIT_BENCH" \
        --base "$BASE" --model "$model" --target "$model" --policy-arm eager \
        --rep 0 --concurrency "$concurrency" --max-tokens 512 --label "$label" \
        --namespace "gscost-$label" --out "$OUT/qualification-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/qualification/$label-metrics.json"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/qualification/$label-server.log"
    echo "QUALIFICATION_PASS model=$model concurrency=$concurrency ts=$(date -u +%FT%TZ)"
}

run_main_point() {
    local model=$1 concurrency=$2 rep=$3 arm=$4
    local label="main-$model-c$concurrency-r$rep-$arm"
    echo "MAIN_POINT_START label=$label ts=$(date -u +%FT%TZ)"
    start_server "$model" "$arm" 4096 "main/$label"
    curl -sf "$BASE/metrics" >"$OUT/main/$label-metrics-before.json"
    snapshot "$OUT/main/$label-thermal-before.log" "$label-before"
    run_logged "$OUT/main/$label-load.log" python3 "$CACHE_HIT_BENCH" \
        --base "$BASE" --model "$model" --target "$model" --policy-arm "$arm" \
        --rep "$rep" --concurrency "$concurrency" --max-tokens 512 --label "$label" \
        --namespace "gscost-$label" --out "$OUT/main-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/main/$label-metrics-after.json"
    snapshot "$OUT/main/$label-thermal-after.log" "$label-after"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/main/$label-server.log"
    echo "MAIN_POINT_PASS label=$label ts=$(date -u +%FT%TZ)"
}

run_mixed_point() {
    local model=$1 concurrency=$2 rep=$3 arm=$4
    local label="mixed-$model-c$concurrency-r$rep-$arm"
    echo "MIXED_POINT_START label=$label ts=$(date -u +%FT%TZ)"
    start_server "$model" "$arm" 4096 "mixed/$label"
    run_warmup "mixed/$label" "$model"
    snapshot "$OUT/mixed/$label-thermal-before.log" "$label-before"
    run_logged "$OUT/mixed/$label-load.log" python3 "$CACHE_BENCH" \
        --base "$BASE" --model "$model" --target "$model" --policy-arm "$arm" \
        --rep "$rep" --concurrency "$concurrency" --label "$label" \
        --namespace "gscost-$label" --out "$OUT/mixed-points.jsonl" \
        --module "$SELLGATE_MODULE" --workload-lock "$WORKLOAD_LOCK" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/mixed/$label-metrics-final.json"
    snapshot "$OUT/mixed/$label-thermal-after.log" "$label-after"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/mixed/$label-server.log"
    echo "MIXED_POINT_PASS label=$label ts=$(date -u +%FT%TZ)"
}

run_battery_logged() {
    local label=$1 timeout_s=$2
    shift 2
    echo "BATTERY_START label=$label ts=$(date -u +%FT%TZ)"
    snapshot "$OUT/battery/$label-gpu-before.log" "$label-before"
    run_logged "$OUT/battery/$label.log" timeout "$timeout_s" "$@"
    wait_idle
    snapshot "$OUT/battery/$label-gpu-after.log" "$label-after"
    echo "BATTERY_DONE label=$label ts=$(date -u +%FT%TZ)"
}

echo "CAMPAIGN_START ts=$(date -u +%FT%TZ) host=$(hostname)"
echo "source=$EXPECTED_SOURCE repo=$REPO out=$OUT"
git -C "$REPO" log -5 --oneline --decorate
python3 - "$BENCH" "$CACHE_HIT_BENCH" "$CACHE_BENCH" "$REDUCE" "$SELLGATE_MODULE" <<'PY'
import ast
import pathlib
import sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
sha256sum "$Q27" "$Q35" "$Q27_DRAFT" "$Q35_DRAFT" "$SERVER" "$KERNEL" \
    "$RUN_GEN" "$RUN_SPEC" "$BENCH" "$CACHE_HIT_BENCH" "$CACHE_BENCH" "$REDUCE" \
    "$SELLGATE_MODULE" "$WORKLOAD_LOCK" "$0" >"$OUT/SHA256SUMS.input"
binary_sha=$(sha256sum "$SERVER" | awk '{print $1}')
echo "$binary_sha  $SERVER" >"$OUT/binary.sha256"
{
    echo "REPAIRED: MEMRA_SERVE_B1FAST=<unset> MEMRA_SERVE_GS=<unset>"
    echo "EAGER: MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1"
    echo "Both arms execute $SERVER"
    echo "Binary SHA-256: $binary_sha"
} >"$OUT/arm-invariant.txt"

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 60 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
lock_acquired=1
lock_start=$(date -u +%FT%TZ)
echo "GPU_LOCK_ACQUIRED ts=$lock_start pid=$$"
assert_idle
assert_port_clear
snapshot "$OUT/gpu-before.log" lock-acquired
nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
sampler_pid=$!

# Non-scored proof that the EAGER environment really instantiates GraphSession for both sold models.
run_activation_probe q27
run_activation_probe q35
# The known defect can turn an EAGER width transition into early EOS.  Qualify the new
# long-form timing prompt at every non-solo Q27 width before accepting any scored cell.
run_length_qualification q27 4
run_length_qualification q27 16
run_length_qualification q27 40

: >"$OUT/main-points.jsonl"
: >"$OUT/warmups.jsonl"
main_cells=(q27:1 q35:4 q27:16 q35:40 q35:1 q27:4 q35:16 q27:40)
for rep in $(seq 1 5); do
    offset=$(( (rep - 1) * 3 % ${#main_cells[@]} ))
    for point in $(seq 0 $((${#main_cells[@]} - 1))); do
        index=$(( (point + offset) % ${#main_cells[@]} ))
        IFS=: read -r model concurrency <<<"${main_cells[$index]}"
        if (( rep % 2 == 1 )); then
            run_main_point "$model" "$concurrency" "$rep" repaired
            run_main_point "$model" "$concurrency" "$rep" eager
        else
            run_main_point "$model" "$concurrency" "$rep" eager
            run_main_point "$model" "$concurrency" "$rep" repaired
        fi
    done
done

: >"$OUT/mixed-points.jsonl"
mixed_cells=(q27:4 q35:40 q27:16 q35:4)
for rep in $(seq 1 5); do
    offset=$(( (rep - 1) % ${#mixed_cells[@]} ))
    for point in $(seq 0 $((${#mixed_cells[@]} - 1))); do
        index=$(( (point + offset) % ${#mixed_cells[@]} ))
        IFS=: read -r model concurrency <<<"${mixed_cells[$index]}"
        if (( rep % 2 == 1 )); then
            run_mixed_point "$model" "$concurrency" "$rep" repaired
            run_mixed_point "$model" "$concurrency" "$rep" eager
        else
            run_mixed_point "$model" "$concurrency" "$rep" eager
            run_mixed_point "$model" "$concurrency" "$rep" repaired
        fi
    done
done

# Pre-release battery, serialized on GPU0 after all scored cells and still under fd 9's lock.
run_battery_logged kernel-check 2400 env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
    CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$MODEL_ROOT" "$KERNEL"
grep -q 'ALL GREEN' "$OUT/battery/kernel-check.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/battery/kernel-check.log"; then
    echo "FAIL: kernel-check emitted a failure verdict"
    exit 1
fi

run_battery_logged run-gen-q27 2400 env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
    CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q27"
grep -q 'argmax=.*MATCH' "$OUT/battery/run-gen-q27.log"
if grep -q 'MISMATCH' "$OUT/battery/run-gen-q27.log"; then
    echo "FAIL: Q27 run-gen emitted MISMATCH"
    exit 1
fi

run_battery_logged run-gen-q35 2400 env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
    CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q35"
grep -q 'argmax=.*MATCH' "$OUT/battery/run-gen-q35.log"
if grep -q 'MISMATCH' "$OUT/battery/run-gen-q35.log"; then
    echo "FAIL: Q35 run-gen emitted MISMATCH"
    exit 1
fi

run_battery_logged run-spec-q27 4800 env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
    -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_SPEC" "$Q27"
test "$(grep -c 'self-consistency: PASS' "$OUT/battery/run-spec-q27.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/battery/run-spec-q27.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/battery/run-spec-q27.log"; then
    echo "FAIL: Q27 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi

run_battery_logged run-spec-q35 4800 env -u MEMRA_SERVE_B1FAST -u MEMRA_SERVE_GS \
    -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_SPEC" "$Q35"
test "$(grep -c 'self-consistency: PASS' "$OUT/battery/run-spec-q35.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/battery/run-spec-q35.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/battery/run-spec-q35.log"; then
    echo "FAIL: Q35 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi

stop_sampler
python3 "$REDUCE" --main "$OUT/main-points.jsonl" --mixed "$OUT/mixed-points.jsonl" \
    --thermal "$OUT/gpu-250ms.csv" --source "$EXPECTED_SOURCE" \
    --binary-sha256 "$binary_sha" --out "$OUT/summary.json" 2>&1 | tee "$OUT/reduce.log"

assert_idle
assert_port_clear
snapshot "$OUT/gpu-after.log" campaign-complete
lock_end=$(date -u +%FT%TZ)
{
    echo "lock_acquired=$lock_start"
    echo "lock_released=$lock_end"
    echo "scored_and_battery_one_hold=true"
} >"$OUT/lock-window.txt"
touch "$OUT/campaign.ok"
echo "CAMPAIGN_PASS ts=$lock_end"

flock -u 9
exec 9>&-
lock_acquired=0
flock -n /tmp/memra-gpu.lock -c 'echo GPU_LOCK_FREE_AFTER_CAMPAIGN'
trap - EXIT INT TERM
echo "CAMPAIGN_CLEAN_EXIT ts=$(date -u +%FT%TZ)"
