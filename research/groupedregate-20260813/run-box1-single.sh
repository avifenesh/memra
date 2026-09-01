#!/usr/bin/env bash
# Single-card grouped-regate campaign: same KAT transfer gate, Q35 exactness, prefill, and serve A/B.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH

ROOT=${GROUPEDREGATE_ROOT:-/opt/scratch/nvme/cx-groupedregate}
REPO=${GROUPEDREGATE_REPO:-$ROOT/memra}
HARNESS=${GROUPEDREGATE_HARNESS:-$ROOT/harness}
MODELS=${GROUPEDREGATE_MODELS:-/opt/scratch/nvme/cx-requal/models}
STAMP=${GROUPEDREGATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${GROUPEDREGATE_OUT:-$ROOT/raw/single-$STAMP}
PHYSICAL_GPU_INDEX=${PHYSICAL_GPU_INDEX:?set the free physical GPU index after live inspection}
N=${N:-5}

EXPECTED_SOURCE=18885ec479d897a3e8c42b0d408a71fa3edaa708
EXPECTED_KAT_SHA=${EXPECTED_KAT_SHA:?set the staged KAT SHA-256}
EXPECTED_Q27=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_Q27_DRAFT=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
EXPECTED_Q35=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
EXPECTED_Q35_DRAFT=ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a
EXPECTED_KAT_PROMPT=dc91551b1e83414616ebb8d65ee88d1af1fc8792dadffe0208b732d094adfc0d
EXPECTED_FROZEN_REPLAY=91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b
EXPECTED_WORKLOAD=85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34

KAT=$ROOT/models/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
Q27=$MODELS/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=$MODELS/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=$MODELS/draft-35b-owntrim-nvfp4head-q4blk.gguf
KAT_PROMPT=$REPO/research/depth-decode-20260802/depth-2048-kat.txt
Q35_PROMPT=$REPO/research/e2e/prompts/board-2048.txt
FROZEN_REPLAY=$REPO/research/sellgate-20260812/sellgate_replay.py
WORKLOAD_LOCK=$REPO/research/sellgate-20260812/workload.lock.json
SERVE_AB=$HARNESS/serve_ab.py

KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
PRIME=$REPO/target/release/concat-prime-probe
SERVER=$REPO/target/release/memra-server
PORT=${GROUPEDREGATE_PORT:-18635}
BASE=http://127.0.0.1:$PORT

test "$N" -ge 5
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"/{exactness,prefill,serve,thermal}
exec > >(tee "$OUT/orchestrator.log") 2>&1

server_pid=
thermal_pid=
host_sampler_pid=
lock_held=0

gpu_uuid() {
    nvidia-smi -i "$PHYSICAL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader
}

selected_apps() {
    local uuid=$1
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F ', *' -v uuid="$uuid" '$1 == uuid { print }'
}

selected_memory_mib() {
    nvidia-smi -i "$PHYSICAL_GPU_INDEX" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' '
}

gpu_fields() {
    nvidia-smi -i "$PHYSICAL_GPU_INDEX" \
        --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        echo "selected_physical_gpu_index=$PHYSICAL_GPU_INDEX"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        echo "compute_apps:"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
            --format=csv,noheader || true
        echo "locks:"
        lslocks -o COMMAND,PID,TYPE,PATH 2>/dev/null \
            | grep -E 'memra-gpu(-1)?[.]lock' || true
        echo "processes:"
        ps -eo user=,pid=,lstart=,args= \
            | grep -E 'cx-cachesize|cx-groupedregate|memra-server|run-gen|run-spec|concat-prime' \
            | grep -v grep || true
    } >"$path" 2>&1
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
    for pid in "${thermal_pid:-}" "${host_sampler_pid:-}"; do
        test -n "$pid" || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    thermal_pid=
    host_sampler_pid=
}

wait_selected_idle() {
    local uuid=$1
    for _ in $(seq 1 120); do
        if test -z "$(selected_apps "$uuid")" && test "$(selected_memory_mib)" -eq 0; then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: selected physical GPU $PHYSICAL_GPU_INDEX did not return to 0 MiB"
    selected_apps "$uuid" || true
    nvidia-smi -i "$PHYSICAL_GPU_INDEX"
    return 1
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

finalize() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_server
    stop_samplers
    local uuid=""
    uuid=$(gpu_uuid 2>/dev/null)
    if test -n "$uuid"; then
        wait_selected_idle "$uuid"
        idle_rc=$?
        test "$idle_rc" -eq 0 || rc=1
    fi
    snapshot "$OUT/shutdown.log" shutdown
    if test "$lock_held" -eq 1; then
        flock -u 9
        lock_held=0
        echo "GROUPEDREGATE_LOCK_RELEASED ts=$(date -u +%FT%TZ)"
    fi
    echo "GROUPEDREGATE_SINGLE_EXIT rc=$rc ts=$(date -u +%FT%TZ)"
    write_manifest
    exit "$rc"
}
trap finalize EXIT INT TERM

run_tee() {
    local label=$1 log=$2 timeout_s=$3
    shift 3
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ) raw=$log"
    set +e
    timeout "$timeout_s" "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$log.exit"
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

append_prefill_row() {
    local model_label=$1 shape=$2 rep=$3 arm=$4 order=$5 log=$6 rc=$7 before=$8 after=$9
    local parsed tokens seconds tok_s
    parsed=$(awk '
        /ppprime MEDIAN:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "MEDIAN:") tokens = $(i + 1)
                if ($i == "in") { seconds = $(i + 1); sub(/s$/, "", seconds) }
                if ($i == "=") rate = $(i + 1)
            }
        }
        END { print tokens, seconds, rate }
    ' "$log")
    read -r tokens seconds tok_s <<<"$parsed"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model_label" "$shape" "$rep" "$arm" "$order" "${tokens:-}" \
        "${seconds:-}" "${tok_s:-}" "$rc" "$PHYSICAL_GPU_INDEX" "$GPU_UUID" \
        "$before|$after" >>"$OUT/prefill/results.tsv"
    test "$rc" -eq 0
    test -n "${tokens:-}" && test -n "${seconds:-}" && test -n "${tok_s:-}"
}

run_prefill() {
    local model_label=$1 shape=$2 model=$3 prompt=$4 rep=$5 arm=$6 order=$7
    local value log before after rc=0
    case "$arm" in
        off) value=0 ;;
        grouped) value=1 ;;
        *) return 2 ;;
    esac
    log="$OUT/prefill/$model_label-$shape-$arm-r$rep.log"
    before=$(gpu_fields)
    snapshot "$OUT/thermal/$model_label-$shape-$arm-r$rep-before.log" \
        "$model_label-$shape-$arm-r$rep-before"
    run_tee "$model_label-$shape-$arm-r$rep" "$log" 3600 \
        env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_CHUNK -u MEMRA_PRIME_PP -u MEMRA_MOE_STATS -u MEMRA_MOE_GATE \
        CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" MEMRA_MOE_GROUPED="$value" \
        nice -n 10 "$PRIME" "$model" ppprime \
        --prompt-a "@$prompt" --reps 1 --warmup 1 || rc=$?
    after=$(gpu_fields)
    snapshot "$OUT/thermal/$model_label-$shape-$arm-r$rep-after.log" \
        "$model_label-$shape-$arm-r$rep-after"
    append_prefill_row "$model_label" "$shape" "$rep" "$arm" "$order" \
        "$log" "$rc" "$before" "$after"
    wait_selected_idle "$GPU_UUID"
}

assert_run_gen() {
    local label=$1 log=$2 require_grouped=$3
    grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$log"
    grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$log"
    if grep -q 'MISMATCH' "$log"; then
        echo "FAIL: $label emitted MISMATCH"
        return 1
    fi
    if test "$require_grouped" -eq 1; then
        grep -q 'dispatch=resident-q8' "$log"
        grep -q 'moe-gate .* BYTE-IDENTICAL' "$log"
    else
        test "$(grep -c 'moe-grouped' "$log" || true)" -eq 0
    fi
}

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

run_serve_boot() {
    local rep=$1 arm=$2 order=$3 value log jsonl client_log before after rc=0 start ready_s
    case "$arm" in
        off) value=0 ;;
        grouped) value=1 ;;
        *) return 2 ;;
    esac
    log="$OUT/serve/server-$arm-r$rep.log"
    jsonl="$OUT/serve/cells-$arm-r$rep.jsonl"
    client_log="$OUT/serve/client-$arm-r$rep.log"
    ss -tln 2>/dev/null | grep -q "[:.]$PORT " \
        && { echo "FAIL: port $PORT already has a listener"; return 1; }
    before=$(gpu_fields)
    snapshot "$OUT/thermal/serve-$arm-r$rep-before.log" "serve-$arm-r$rep-before"
    start=$(date +%s)
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST \
        -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" MEMRA_MOE_GROUPED="$value" \
        MEMRA_MODELS="q35=$Q35" MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_TAG="cx-groupedregate-$arm-r$rep" MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" > >(tee "$log") 2>&1 &
    server_pid=$!
    wait_ready "$log"
    ready_s=$(( $(date +%s) - start ))
    curl -sf "$BASE/v1/models" >"$OUT/serve/models-$arm-r$rep.json"
    curl -sf "$BASE/metrics" >"$OUT/serve/metrics-before-$arm-r$rep.json"
    run_tee "serve-$arm-r$rep" "$client_log" 21600 \
        python3 "$SERVE_AB" --endpoint "q35,$BASE,q35" \
        --frozen-replay "$FROZEN_REPLAY" --workload-lock "$WORKLOAD_LOCK" \
        --out "$jsonl" --namespace "cx-groupedregate-$arm-r$rep" \
        --dispatch-arm "$arm" --repetition "$rep" \
        --physical-gpu-index "$PHYSICAL_GPU_INDEX" --gpu-uuid "$GPU_UUID" \
        --source-commit "$EXPECTED_SOURCE" --timeout 1800 || rc=$?
    curl -sf "$BASE/metrics" >"$OUT/serve/metrics-after-$arm-r$rep.json"
    stop_server
    assert_server_clean "$log"
    wait_selected_idle "$GPU_UUID"
    after=$(gpu_fields)
    snapshot "$OUT/thermal/serve-$arm-r$rep-after.log" "serve-$arm-r$rep-after"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rep" "$arm" "$order" "$rc" "$ready_s" "$PHYSICAL_GPU_INDEX" \
        "$GPU_UUID" "$before" "$after" >>"$OUT/serve/boots.tsv"
    test "$rc" -eq 0
    grep -q '"kind": "summary".*"verdict": "PASS"' "$jsonl"
}

echo "GROUPEDREGATE_SINGLE_START ts=$(date -u +%FT%TZ) pid=$$ source=$EXPECTED_SOURCE"
cd "$REPO"
test "$(hostname)" = <private-host-redacted>
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test "$(git describe --tags --exact-match HEAD)" = v0.81.2
test -z "$(git status --porcelain --untracked-files=all)"
test "$PHYSICAL_GPU_INDEX" -eq 0 -o "$PHYSICAL_GPU_INDEX" -eq 1
OTHER_GPU_INDEX=$((1 - PHYSICAL_GPU_INDEX))
GPU_UUID=$(gpu_uuid)
OTHER_GPU_UUID=$(nvidia-smi -i "$OTHER_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)

for binary in "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$PRIME" "$SERVER"; do
    test -x "$binary"
done
test -f "$SERVE_AB"

{
    echo "source_commit=$EXPECTED_SOURCE"
    echo "source_describe=$(git describe --always --dirty)"
    echo "host=$(hostname)"
    echo "start_ts=$(date -u +%FT%TZ)"
    echo "coordination_lock=/tmp/memra-gpu-1.lock"
    echo "selected_physical_gpu_index=$PHYSICAL_GPU_INDEX"
    echo "selected_gpu_uuid=$GPU_UUID"
    echo "co_tenant_physical_gpu_index=$OTHER_GPU_INDEX"
    echo "co_tenant_gpu_uuid=$OTHER_GPU_UUID"
    echo "protocol=N=$N per arm; adjacent interleaved pairs; order alternated; one warmup plus one timed prime per independent process; one flock hold"
    uname -a
    rustc --version
    cargo --version
    nvcc --version
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
    git status --short --branch --untracked-files=all
    check_hash "$EXPECTED_KAT_SHA" "$KAT"
    check_hash "$EXPECTED_Q27" "$Q27"
    check_hash "$EXPECTED_Q27_DRAFT" "$Q27_DRAFT"
    check_hash "$EXPECTED_Q35" "$Q35"
    check_hash "$EXPECTED_Q35_DRAFT" "$Q35_DRAFT"
    check_hash "$EXPECTED_KAT_PROMPT" "$KAT_PROMPT"
    check_hash "$EXPECTED_FROZEN_REPLAY" "$FROZEN_REPLAY"
    check_hash "$EXPECTED_WORKLOAD" "$WORKLOAD_LOCK"
    sha256sum "$Q35_PROMPT" "$SERVE_AB" "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$PRIME" "$SERVER"
    stat -c 'artifact=%n bytes=%s mtime=%y' "$KAT" "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT"
} 2>&1 | tee "$OUT/provenance.log"

snapshot "$OUT/coordination-before-lock.log" coordination-before-lock
test -z "$(selected_apps "$GPU_UUID")" \
    || { selected_apps "$GPU_UUID"; echo "FAIL: selected physical GPU is busy"; exit 76; }
test "$(selected_memory_mib)" -eq 0 \
    || { echo "FAIL: selected physical GPU is not at 0 MiB"; exit 76; }

exec 9>/tmp/memra-gpu-1.lock
flock -w 14400 9 || { echo "FAIL: GPU-1 lock timeout"; exit 75; }
lock_held=1
echo "GROUPEDREGATE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
snapshot "$OUT/gpu-before.log" before
test -z "$(selected_apps "$GPU_UUID")" \
    || { selected_apps "$GPU_UUID"; echo "FAIL: selected physical GPU became busy"; exit 76; }
test "$(selected_memory_mib)" -eq 0

nvidia-smi -i "$PHYSICAL_GPU_INDEX" \
    --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -lms 250 >"$OUT/thermal/selected-gpu-250ms.csv" 2>&1 &
thermal_pid=$!
nvidia-smi \
    --query-gpu=timestamp,index,uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 >"$OUT/thermal/both-gpus-1s.csv" 2>&1 &
host_sampler_pid=$!

printf 'model\tshape\trep\tarm\torder\ttokens\tseconds\ttok_s\trc\tphysical_gpu_index\tgpu_uuid\tthermal_before_after\n' \
    >"$OUT/prefill/results.tsv"
order=0
for rep in $(seq 1 "$N"); do
    if (( rep % 2 == 1 )); then
        arms=(off grouped)
    else
        arms=(grouped off)
    fi
    for arm in "${arms[@]}"; do
        order=$((order + 1))
        run_prefill kat pp2048 "$KAT" "$KAT_PROMPT" "$rep" "$arm" "$order"
    done
done

run_tee kernel-check-kat "$OUT/exactness/kernel-check-kat.log" 3600 \
    env -u MEMRA_KC_FAST -u MEMRA_KC_ONLY CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=1 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL" "$KAT" \
    --require-manifest tools/kernel-check-27b.cells \
    --require-manifest tools/kernel-check-step35.cells
grep -q '^ALL GREEN (' "$OUT/exactness/kernel-check-kat.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/exactness/kernel-check-kat.log"; then
    echo "FAIL: KAT kernel-check emitted a failure marker"
    exit 1
fi
wait_selected_idle "$GPU_UUID"

run_tee kernel-check-q35 "$OUT/exactness/kernel-check-q35.log" 3600 \
    env -u MEMRA_KC_FAST -u MEMRA_KC_ONLY CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=1 MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL" "$Q35" \
    --require-manifest tools/kernel-check-27b.cells \
    --require-manifest tools/kernel-check-step35.cells
grep -q '^ALL GREEN (' "$OUT/exactness/kernel-check-q35.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/exactness/kernel-check-q35.log"; then
    echo "FAIL: Q35 kernel-check emitted a failure marker"
    exit 1
fi
wait_selected_idle "$GPU_UUID"

run_tee run-gen-kat-grouped "$OUT/exactness/run-gen-kat-grouped.log" 3600 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=1 MEMRA_MOE_GATE=1 MEMRA_MOE_STATS=1 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$KAT_PROMPT" "$RUN_GEN" "$KAT"
assert_run_gen run-gen-kat-grouped "$OUT/exactness/run-gen-kat-grouped.log" 1
wait_selected_idle "$GPU_UUID"

run_tee run-gen-kat-off "$OUT/exactness/run-gen-kat-off.log" 3600 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=0 MEMRA_MOE_STATS=1 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$KAT_PROMPT" "$RUN_GEN" "$KAT"
assert_run_gen run-gen-kat-off "$OUT/exactness/run-gen-kat-off.log" 0
wait_selected_idle "$GPU_UUID"

run_tee run-gen-q35-grouped "$OUT/exactness/run-gen-q35-grouped.log" 3600 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=1 MEMRA_MOE_GATE=1 MEMRA_MOE_STATS=1 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$REPO/research/e2e/prompts/pp512.txt" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q35"
assert_run_gen run-gen-q35-grouped "$OUT/exactness/run-gen-q35-grouped.log" 1
wait_selected_idle "$GPU_UUID"

run_tee run-gen-q35-off "$OUT/exactness/run-gen-q35-off.log" 3600 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
    MEMRA_MOE_GROUPED=0 MEMRA_MOE_STATS=1 MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$REPO/research/e2e/prompts/pp512.txt" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q35"
assert_run_gen run-gen-q35-off "$OUT/exactness/run-gen-q35-off.log" 0
wait_selected_idle "$GPU_UUID"

run_tee run-spec-q35 "$OUT/exactness/run-spec-q35.log" 7200 \
    env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES \
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" MEMRA_MOE_GROUPED=1 \
    MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$REPO/research/e2e/prompts/pp512.txt" MEMRA_CHAT=1 \
    "$RUN_SPEC" "$Q35"
test "$(grep -c 'self-consistency: PASS' "$OUT/exactness/run-spec-q35.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/exactness/run-spec-q35.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/exactness/run-spec-q35.log"; then
    echo "FAIL: Q35 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi
wait_selected_idle "$GPU_UUID"

run_tee serve-smoke "$OUT/exactness/serve-smoke.log" 14400 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
    -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_PRIME_BATCH_MAX_T \
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" MEMRA_MOE_GROUPED=1 \
    MEMRA_Q35_COLD_MODEL="$Q35" bash tools/serve-smoke.sh "$Q27" "$Q27_DRAFT"
test -f /tmp/serve-smoke.log \
    && cp /tmp/serve-smoke.log "$OUT/exactness/serve-smoke-final-server.log"
test -f /tmp/serve-smoke-q35-cold-mixed.log \
    && cp /tmp/serve-smoke-q35-cold-mixed.log "$OUT/exactness/serve-smoke-q35-cell.jsonl"
grep -q 'Q35 mixed c=4: 20/20 requests reached exactly 60 tokens' \
    "$OUT/exactness/serve-smoke.log"
grep -q 'serve-smoke: 0 failed' "$OUT/exactness/serve-smoke.log"
wait_selected_idle "$GPU_UUID"
touch "$OUT/exactness/exactness.ok"

order=0
for rep in $(seq 1 "$N"); do
    if (( rep % 2 == 1 )); then
        arms=(off grouped)
    else
        arms=(grouped off)
    fi
    for arm in "${arms[@]}"; do
        order=$((order + 1))
        run_prefill q35 board2048 "$Q35" "$Q35_PROMPT" "$rep" "$arm" "$order"
    done
done

printf 'rep\tarm\torder\trc\tready_s\tphysical_gpu_index\tgpu_uuid\tthermal_before\tthermal_after\n' \
    >"$OUT/serve/boots.tsv"
order=0
for rep in $(seq 1 "$N"); do
    if (( rep % 2 == 1 )); then
        arms=(off grouped)
    else
        arms=(grouped off)
    fi
    for arm in "${arms[@]}"; do
        order=$((order + 1))
        run_serve_boot "$rep" "$arm" "$order"
    done
done

touch "$OUT/single.complete"
echo "GROUPEDREGATE_SINGLE_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT"
