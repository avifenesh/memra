#!/usr/bin/env bash
# Full cx-selfserve serving battery for box1 (2x RTX PRO 6000 Blackwell).
set -Eeuo pipefail

repo=${SELFSERVE_REPO:-/opt/dl-image/nvme/cx-selfserve/memra}
out=${SELFSERVE_OUT:-$repo/research/selfserve-20260813/raw/box1-$(date -u +%Y%m%dT%H%M%SZ)}
source_revision=${SELFSERVE_SOURCE_REVISION:-unknown}
models=${SELFSERVE_MODELS_DIR:-/opt/dl-image/nvme/cx-requal/models}
q27=${SELFSERVE_Q27:-$models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
q27_draft=${SELFSERVE_Q27_DRAFT:-$models/draft-daily-owntrim-nvfp4head-q4blk.gguf}
q35=${SELFSERVE_Q35:-$models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
q35_draft=${SELFSERVE_Q35_DRAFT:-$models/draft-35b-owntrim-nvfp4head-q4blk.gguf}
prompt=${SELFSERVE_PROMPT:-$repo/research/e2e/prompts/pp512.txt}
gpu_lock_wait_s=${SELFSERVE_GPU_LOCK_WAIT_S:-43200}
events=$out/events.log
serve_smoke_marker=$out/serve-smoke.started

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin
export CUDA_HOME=/usr/local/cuda-13.2
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

mkdir -p "$out"
cd "$repo"

event() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$events"
}

fail() {
    event "FAIL $*"
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

run_logged() {
    local name=$1
    local log=$2
    shift 2
    event "START $name"
    set +e
    "$@" >"$log" 2>&1
    local rc=$?
    set -e
    printf '%d\n' "$rc" >"$out/$name.exit"
    event "END $name rc=$rc"
    return "$rc"
}

collect_serve_smoke_logs() {
    local path name
    while IFS= read -r -d '' path; do
        name=${path##*/}
        cp -f -- "$path" "$out/tmp-$name"
    done < <(find /tmp -maxdepth 1 -type f -name 'serve-smoke*.log' \
        -newer "$serve_smoke_marker" -print0 2>/dev/null)
}

budget_server_pid=
budget_state=
gpu_monitor_pid=
foreign_gpu_log=$out/foreign-gpu-processes.log
vanished_gpu_log=$out/vanished-gpu-processes.log

stop_gpu_monitor() {
    if [[ -z ${gpu_monitor_pid:-} ]]; then
        return 0
    fi
    kill -TERM "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
    gpu_monitor_pid=
}

monitor_foreign_gpu_processes() {
    local row pid process memory executable state
    while true; do
        while IFS= read -r row; do
            [[ -n $row ]] || continue
            IFS=, read -r pid process memory <<<"$row"
            pid=${pid#"${pid%%[![:space:]]*}"}
            process=${process#"${process%%[![:space:]]*}"}
            memory=${memory#"${memory%%[![:space:]]*}"}
            executable=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
            state=$(awk '/^State:/ { print $2; exit }' "/proc/$pid/status" \
                2>/dev/null || true)
            if [[ -z $executable && ( ! -d /proc/$pid || $state == Z || $state == X ) ]]; then
                printf '%s pid=%s process=%s state=%s used_memory=%s\n' \
                    "$(date -u +%FT%TZ)" "$pid" "$process" \
                    "${state:-vanished}" "$memory" \
                    >>"$vanished_gpu_log"
                continue
            fi
            case "$executable" in
                "$repo"/target/release/*) ;;
                *)
                    printf '%s pid=%s process=%s executable=%s used_memory=%s\n' \
                        "$(date -u +%FT%TZ)" "$pid" "$process" \
                        "${executable:-unresolved}" "$memory" \
                        >>"$foreign_gpu_log"
                    return 0
                    ;;
            esac
        done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>>"$out/foreign-gpu-monitor-errors.log" || true)
        sleep 1
    done
}

stop_budget_server() {
    if [[ -z ${budget_server_pid:-} ]]; then
        return 0
    fi
    local state=
    if [[ -r /proc/$budget_server_pid/stat ]]; then
        state=$(awk '{print $3}' "/proc/$budget_server_pid/stat")
    fi
    if kill -0 "$budget_server_pid" 2>/dev/null && [[ $state != Z ]]; then
        kill -TERM "$budget_server_pid" 2>/dev/null || true
        for _ in $(seq 1 300); do
            if ! kill -0 "$budget_server_pid" 2>/dev/null; then
                break
            fi
            state=$(awk '{print $3}' "/proc/$budget_server_pid/stat" 2>/dev/null || true)
            if [[ $state == Z ]]; then
                break
            fi
            sleep 0.1
        done
    fi
    state=$(awk '{print $3}' "/proc/$budget_server_pid/stat" 2>/dev/null || true)
    if kill -0 "$budget_server_pid" 2>/dev/null && [[ $state != Z ]]; then
        return 1
    fi
    wait "$budget_server_pid" 2>/dev/null || true
    budget_server_pid=
}

clean_budget_state() {
    if [[ -z ${budget_state:-} ]]; then
        return 0
    fi
    case "$budget_state" in
        /tmp/memra-selfserve-budget.*) rm -rf -- "$budget_state" ;;
        *) printf 'refusing to remove unexpected budget state path: %s\n' "$budget_state" >&2; return 1 ;;
    esac
    budget_state=
}

cleanup() {
    stop_gpu_monitor || true
    stop_budget_server || true
    clean_budget_state || true
}
trap cleanup EXIT

run_budget_smoke() {
    local public_addr=127.0.0.1:8185
    local admin_addr=127.0.0.1:8005
    local smoke_out=$out/budget-smoke
    mkdir -p "$smoke_out"
    budget_state=$(mktemp -d /tmp/memra-selfserve-budget.XXXXXX)
    umask 077
    : >"$budget_state/keys.toml"
    chmod 0640 "$budget_state/keys.toml"
    printf '%s\n' \
        '[[budgets]]' \
        'tenant = "smoke"' \
        'currency = "USD"' \
        'balance_micro = 10000' >"$budget_state/budgets.toml"
    chmod 0640 "$budget_state/budgets.toml"
    printf '%s\n' \
        '[models."smoke".pricing]' \
        'prompt = "0.000000001"' \
        'cached_prompt = "0"' \
        'completion = "0.000001"' \
        'request = "0.009998"' >"$budget_state/models.toml"
    chmod 0640 "$budget_state/models.toml"
    python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_hex(32) + "\n")' \
        >"$budget_state/admin-token"
    chmod 0600 "$budget_state/admin-token"

    event 'START budget-server'
    CUDA_VISIBLE_DEVICES=0 \
        MEMRA_COMPAT=openai \
        MEMRA_MODELS="smoke=$q27" \
        MEMRA_MODEL_METADATA="$budget_state/models.toml" \
        MEMRA_ADDR="$public_addr" \
        MEMRA_API_KEYS="$budget_state/keys.toml" \
        MEMRA_REQUEST_LEDGER="$budget_state/requests.jsonl" \
        MEMRA_TENANT_BUDGETS="$budget_state/budgets.toml" \
        MEMRA_ADMIN_ADDR="$admin_addr" \
        MEMRA_ADMIN_TOKEN_FILE="$budget_state/admin-token" \
        MEMRA_SERVE_SPEC=0 \
        target/release/memra-server >"$smoke_out/server.log" 2>&1 &
    budget_server_pid=$!
    local ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://$public_addr/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$budget_server_pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [[ $ready -ne 1 ]]; then
        stop_budget_server || true
        return 1
    fi

    set +e
    timeout 1800 python3 research/selfserve-20260813/budget-smoke.py \
        --public-base "http://$public_addr" \
        --admin-base "http://$admin_addr" \
        --admin-token-file "$budget_state/admin-token" \
        --state-dir "$budget_state" \
        --out "$smoke_out" >"$smoke_out/client.log" 2>&1
    local client_rc=$?
    set -e
    local stop_rc=0
    stop_budget_server || stop_rc=$?
    event "END budget-server client_rc=$client_rc stop_rc=$stop_rc"
    clean_budget_state
    [[ $client_rc -eq 0 && $stop_rc -eq 0 ]]
}

event "BEGIN out=$out source_revision=$source_revision"
{
    printf 'source_revision=%s\n' "$source_revision"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'hostname=%s\n' "$(hostname)"
    uname -a
    rustc --version
    cargo --version
    nvcc --version
} >"$out/provenance.txt" 2>&1

for required in "$q27" "$q27_draft" "$q35" "$q35_draft" "$prompt"; do
    [[ -f $required ]] || fail "missing required file: $required"
done

nvidia-smi -q >"$out/nvidia-before-lock.log" 2>&1
event 'WAIT GPU_LOCK'
exec 9>/tmp/memra-gpu.lock
flock -w "$gpu_lock_wait_s" 9 || fail "GPU lock timeout after ${gpu_lock_wait_s}s"
event 'GPU_LOCK_ACQUIRED'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"$out/compute-apps-before.txt" 2>&1
[[ ! -s $out/compute-apps-before.txt ]] || fail 'GPU was not tenant-clean after lock acquisition'
pgrep -ax memra-server >"$out/memra-processes-before.txt" 2>&1 || true
[[ ! -s $out/memra-processes-before.txt ]] \
    || fail 'memra-server process remained after lock acquisition'
: >"$foreign_gpu_log"
: >"$vanished_gpu_log"
monitor_foreign_gpu_processes &
gpu_monitor_pid=$!

if ! run_logged build "$out/build.log" timeout 5400 cargo build --release; then
    fail 'release build failed'
fi

sha256sum "$q27" "$q27_draft" "$q35" "$q35_draft" "$prompt" \
    target/release/kernel-check target/release/run-gen target/release/run-spec \
    target/release/memra-server >"$out/SHA256SUMS"

if ! run_logged kernel-check "$out/kernel-check.log" \
    timeout 3600 env CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$models" \
        target/release/kernel-check; then
    fail 'kernel-check exited nonzero'
fi
grep -q 'ALL GREEN' "$out/kernel-check.log" || fail 'kernel-check omitted ALL GREEN'
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$out/kernel-check.log"; then
    fail 'kernel-check emitted a failure verdict'
fi
event 'PASS kernel-check'

if ! run_logged run-gen-q27 "$out/run-gen-q27.log" \
    timeout 3600 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 target/release/run-gen "$q27"; then
    fail 'run-gen q27 exited nonzero'
fi
[[ $(grep -c 'MATCH' "$out/run-gen-q27.log") -ge 2 ]] || fail 'run-gen q27 omitted both MATCH verdicts'
! grep -q 'MISMATCH' "$out/run-gen-q27.log" || fail 'run-gen q27 emitted MISMATCH'
event 'PASS run-gen-q27'

if ! run_logged run-gen-q35 "$out/run-gen-q35.log" \
    timeout 3600 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 target/release/run-gen "$q35"; then
    fail 'run-gen q35 exited nonzero'
fi
[[ $(grep -c 'MATCH' "$out/run-gen-q35.log") -ge 2 ]] || fail 'run-gen q35 omitted both MATCH verdicts'
! grep -q 'MISMATCH' "$out/run-gen-q35.log" || fail 'run-gen q35 emitted MISMATCH'
event 'PASS run-gen-q35'

if ! run_logged run-spec-q27 "$out/run-spec-q27.log" \
    timeout 5400 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
        MEMRA_MTP_DRAFT="$q27_draft" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 target/release/run-spec "$q27"; then
    fail 'run-spec q27 exited nonzero'
fi
[[ $(grep -c 'self-consistency: PASS' "$out/run-spec-q27.log") -eq 8 ]] \
    || fail 'run-spec q27 did not pass every K=1..8'
grep -q '=== SELF-CONSISTENCY PASS ===' "$out/run-spec-q27.log" \
    || fail 'run-spec q27 omitted overall PASS'
! grep -q 'SELF-CONSISTENCY FAIL' "$out/run-spec-q27.log" \
    || fail 'run-spec q27 emitted a failure verdict'
event 'PASS run-spec-q27'

if ! run_logged run-spec-q35 "$out/run-spec-q35.log" \
    timeout 5400 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
        MEMRA_MTP_DRAFT="$q35_draft" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$prompt" \
        MEMRA_CHAT=1 target/release/run-spec "$q35"; then
    fail 'run-spec q35 exited nonzero'
fi
[[ $(grep -c 'self-consistency: PASS' "$out/run-spec-q35.log") -eq 8 ]] \
    || fail 'run-spec q35 did not pass every K=1..8'
grep -q '=== SELF-CONSISTENCY PASS ===' "$out/run-spec-q35.log" \
    || fail 'run-spec q35 omitted overall PASS'
! grep -q 'SELF-CONSISTENCY FAIL' "$out/run-spec-q35.log" \
    || fail 'run-spec q35 emitted a failure verdict'
event 'PASS run-spec-q35'

touch "$serve_smoke_marker"
if ! run_logged serve-smoke "$out/serve-smoke.log" \
    timeout 10800 env CUDA_VISIBLE_DEVICES=0 MEMRA_Q35_COLD_MODEL="$q35" \
        GEMMA_MODEL=/nonexistent/selfserve-gemma.gguf \
        tools/serve-smoke.sh "$q27" "$q27_draft"; then
    collect_serve_smoke_logs
    fail 'serve-smoke exited nonzero'
fi
collect_serve_smoke_logs
grep -q '^serve-smoke: 0 failed$' "$out/serve-smoke.log" \
    || fail 'serve-smoke did not report 0 failed'
event 'PASS serve-smoke'

if ! run_logged serve-stress "$out/serve-stress.log" \
    timeout 7200 env CUDA_VISIBLE_DEVICES=0 \
        MEMRA_STRESS_LOG="$out/serve-stress-server.log" \
        MEMRA_STRESS_ROWS="$out/serve-stress-rows.jsonl" \
        tools/serve-stress-gate.sh "$q27" "$q27_draft" 64; then
    fail 'serve-stress c=64 exited nonzero'
fi
grep -q 'serve-stress-gate: ALL GREEN (c=64' "$out/serve-stress.log" \
    || fail 'serve-stress c=64 omitted ALL GREEN'
event 'PASS serve-stress-c64'

if ! run_logged budget-smoke "$out/budget-smoke-console.log" run_budget_smoke; then
    fail 'one-cent budget-enforcement smoke failed'
fi
grep -q '^budget-smoke: PASS ' "$out/budget-smoke/client.log" \
    || fail 'budget smoke omitted PASS receipt'
event 'PASS budget-smoke'

stop_gpu_monitor
[[ ! -s $foreign_gpu_log ]] \
    || fail 'foreign GPU process was observed while the box1 lock was held'
nvidia-smi -q >"$out/nvidia-after.log" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"$out/compute-apps-after.txt" 2>&1
[[ ! -s $out/compute-apps-after.txt ]] || fail 'GPU processes remained after tenant-clean shutdown'
pgrep -ax memra-server >"$out/memra-processes-after.txt" 2>&1 || true
[[ ! -s $out/memra-processes-after.txt ]] \
    || fail 'memra-server process remained after tenant-clean shutdown'

event 'PASS all-gates'
touch "$out/gates.ok"
(
    cd "$out"
    find . -type f ! -name 'MANIFEST.sha256' ! -name launcher.log -print0 \
        | sort -z | xargs -0 sha256sum
) >"$out/MANIFEST.sha256"
printf 'selfserve box1 battery: PASS\nraw=%s\n' "$out"
