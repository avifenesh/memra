#!/usr/bin/env bash
# v0.81.1 coldship pre-release battery for the designated box1 2x RTX PRO 6000 rig.
# One model uses GPU0 at a time; the global lock reserves and observes both GPUs.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH

REPO=${COLDSHIP_REPO:-/opt/dl-image/nvme/cx-coldship/memra}
EXPECTED_SOURCE=${COLDSHIP_EXPECTED_SOURCE:?set COLDSHIP_EXPECTED_SOURCE}
MODELS=${COLDSHIP_MODELS:-/opt/dl-image/nvme/cx-requal/models}
OUT=${COLDSHIP_OUT:-$REPO/research/coldfix-20260812/raw/coldship-box1}

Q27=$MODELS/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=$MODELS/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=$MODELS/draft-35b-owntrim-nvfp4head-q4blk.gguf
PROMPT=$REPO/research/e2e/prompts/pp512.txt

EXPECTED_Q27=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_Q27_DRAFT=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
EXPECTED_Q35=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
EXPECTED_Q35_DRAFT=ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a

KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
SERVER=$REPO/target/release/memra-server

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

thermal_pid=
lock_held=0

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
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } 2>&1 | tee "$path"
}

wait_idle() {
    local _ apps
    for _ in $(seq 1 180); do
        apps=$(compute_apps)
        test -z "$apps" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

stop_sampler() {
    test -n "${thermal_pid:-}" || return 0
    kill "$thermal_pid" 2>/dev/null || true
    wait "$thermal_pid" 2>/dev/null || true
    thermal_pid=
}

stop_owned_servers() {
    local pid cwd
    while read -r pid; do
        test -n "$pid" || continue
        cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
        if test "$cwd" = "$REPO"; then
            echo "cleanup: stopping owned memra-server pid=$pid cwd=$cwd"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done < <(pgrep -x memra-server || true)
    for _ in $(seq 1 120); do
        local owned=0
        while read -r pid; do
            test -n "$pid" || continue
            cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
            test "$cwd" = "$REPO" && owned=$((owned + 1))
        done < <(pgrep -x memra-server || true)
        test "$owned" -eq 0 && return 0
        sleep 1
    done
    return 1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_sampler
    stop_owned_servers || true
    if test "$lock_held" -eq 1; then
        {
            echo "cleanup_ts=$(date -u +%FT%TZ) rc=$rc"
            echo "compute_apps:"
            compute_apps
            echo "listeners:"
            ss -H -ltnp 2>/dev/null | grep -E ':(8177|8179|18427|18435)\b' || true
        } 2>&1 | tee -a "$OUT/cleanup.log"
        flock -u 9 || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

run_logged() {
    local label=$1 log=$2 timeout_s=$3
    shift 3
    local rc
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ) n=1"
    set +e
    timeout "$timeout_s" "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/$label.exit"
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

cd "$REPO"
test "$(hostname)" = <private-host-redacted>
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e target || { echo "FAIL: target exists; release build would not be fresh"; exit 2; }
for path in "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$PROMPT" \
            tools/kernel-check-27b.cells tools/kernel-check-step35.cells; do
    test -f "$path" || { echo "FAIL: missing battery input: $path"; exit 2; }
done

{
    echo "source_commit=$(git rev-parse HEAD)"
    echo "source_describe=$(git describe --always --dirty)"
    echo "host=$(hostname)"
    echo "start_ts=$(date -u +%FT%TZ)"
    uname -a
    rustc --version
    cargo --version
    nvcc --version
    git status --short --branch --untracked-files=no
    grep -n '^version = "0\.81\.1"$' Cargo.toml
    grep -n 'version = "=0\.81\.1"' Cargo.toml
    check_hash "$EXPECTED_Q27" "$Q27"
    check_hash "$EXPECTED_Q27_DRAFT" "$Q27_DRAFT"
    check_hash "$EXPECTED_Q35" "$Q35"
    check_hash "$EXPECTED_Q35_DRAFT" "$Q35_DRAFT"
    sha256sum "$PROMPT" tools/kernel-check-27b.cells tools/kernel-check-step35.cells \
        tools/serve-smoke.sh tools/serve-stress-gate.sh tools/q35-cold-mixed-gate.py
    stat -c 'artifact=%n bytes=%s mtime=%y' "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT"
} 2>&1 | tee "$OUT/provenance.log"

exec 9>/tmp/memra-gpu.lock
flock -w 300 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
lock_held=1
echo "COLDSHIP_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
snapshot "$OUT/gpu-before.log" before
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle after lock acquisition"; exit 76; }

nvidia-smi \
    --query-gpu=timestamp,index,name,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -lms 1000 >"$OUT/gpu-1s.csv" 2>&1 &
thermal_pid=$!

run_logged build "$OUT/build.log" 7200 cargo build --release

cargo metadata --locked --no-deps --format-version 1 \
    | jq -r '.packages[] | select(.name != "memra-probe") | [.name,.version] | @tsv' \
    | sort | tee "$OUT/workspace-versions.tsv"
test "$(awk '$2 != "0.81.1" { bad++ } END { print bad+0 }' "$OUT/workspace-versions.tsv")" -eq 0
test "$(wc -l <"$OUT/workspace-versions.tsv")" -eq 9

sha256sum "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" \
    | tee "$OUT/runtime-binaries.sha256"

run_logged kernel-check "$OUT/kernel-check.log" 3600 \
    env -u MEMRA_KC_FAST -u MEMRA_KC_ONLY CUDA_VISIBLE_DEVICES=0 \
    MEMRA_KC_MODELS_DIR="$MODELS" "$KERNEL" \
    --require-manifest tools/kernel-check-27b.cells \
    --require-manifest tools/kernel-check-step35.cells
grep -q '^ALL GREEN (' "$OUT/kernel-check.log"
if grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-check.log"; then
    echo "FAIL: kernel-check emitted a failure marker"
    exit 1
fi
wait_idle

run_logged run-gen-q27 "$OUT/run-gen-q27.log" 3600 \
    env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q27"
grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen-q27.log"
grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/run-gen-q27.log"
if grep -q 'MISMATCH' "$OUT/run-gen-q27.log"; then
    echo "FAIL: Q27 run-gen emitted MISMATCH"
    exit 1
fi
wait_idle

run_logged run-gen-q35 "$OUT/run-gen-q35.log" 3600 \
    env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_GEN" "$Q35"
grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen-q35.log"
grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/run-gen-q35.log"
if grep -q 'MISMATCH' "$OUT/run-gen-q35.log"; then
    echo "FAIL: Q35 run-gen emitted MISMATCH"
    exit 1
fi
wait_idle

run_logged run-spec-q27 "$OUT/run-spec-q27.log" 7200 \
    env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_SPEC" "$Q27"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q27.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q27.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q27.log"; then
    echo "FAIL: Q27 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi
wait_idle

run_logged run-spec-q35 "$OUT/run-spec-q35.log" 7200 \
    env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_SPEC" "$Q35"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q35.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q35.log"
if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q35.log"; then
    echo "FAIL: Q35 run-spec emitted SELF-CONSISTENCY FAIL"
    exit 1
fi
wait_idle

smoke_rc=0
run_logged serve-smoke "$OUT/serve-smoke.log" 14400 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
    -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_PRIME_BATCH_MAX_T \
    CUDA_VISIBLE_DEVICES=0 MEMRA_Q35_COLD_MODEL="$Q35" \
    bash tools/serve-smoke.sh "$Q27" "$Q27_DRAFT" || smoke_rc=$?
test -f /tmp/serve-smoke.log && cp /tmp/serve-smoke.log "$OUT/serve-smoke-q35-server.log"
test -f /tmp/serve-smoke-q35-cold-mixed.log \
    && cp /tmp/serve-smoke-q35-cold-mixed.log "$OUT/serve-smoke-q35-cell.jsonl"
test "$smoke_rc" -eq 0
grep -q 'Q35 mixed c=4: 20/20 requests reached exactly 60 tokens' "$OUT/serve-smoke.log"
grep -q 'Q35 routed-MoE carried prime batches remain gated' "$OUT/serve-smoke.log"
grep -q 'serve-smoke: 0 failed' "$OUT/serve-smoke.log"

python3 - "$OUT/serve-smoke-q35-cell.jsonl" <<'PY' | tee "$OUT/q35-cell-verdict.txt"
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
gate = [row for row in rows if row.get("kind") == "q35_cold_mixed_gate"]
cell = [row for row in rows if row.get("kind") == "cell"]
assert len(gate) == 1 and len(cell) == 1
assert gate[0]["verdict"] == "PASS"
assert gate[0]["requests"] == 20 and gate[0]["expected_completion_tokens"] == 60
assert not gate[0]["short_or_non_length"] and not gate[0]["seed_failures"]
assert cell[0]["requests_n"] == 20 and cell[0]["requests_ok"] == 20
assert cell[0]["completion_tokens"] == 1200 and cell[0]["clean"] is True
print("q35_mixed_c4=PASS requests=20/20 exact_tokens=60 completion_tokens=1200")
PY

carried=$(grep -Ec '^\[prime-batch\].*carried=[1-9]' "$OUT/serve-smoke-q35-server.log" || true)
echo "q35_routed_moe_carried_prime_batch_lines=$carried" | tee "$OUT/q35-carried-count.txt"
test "$carried" -eq 0
wait_idle

stress_rc=0
run_logged serve-stress-c64 "$OUT/serve-stress.log" 14400 \
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
    CUDA_VISIBLE_DEVICES=0 MEMRA_STRESS_LOG="$OUT/serve-stress-server.log" \
    MEMRA_STRESS_ROWS="$OUT/serve-stress-rows.jsonl" \
    bash tools/serve-stress-gate.sh "$Q27" "$Q27_DRAFT" 64 || stress_rc=$?
test "$stress_rc" -eq 0
grep -q 'completed 64/64' "$OUT/serve-stress.log"
grep -q 'serve-stress-gate: ALL GREEN' "$OUT/serve-stress.log"
test "$(wc -l <"$OUT/serve-stress-rows.jsonl")" -eq 64
python3 - "$OUT/serve-stress-rows.jsonl" <<'PY' | tee "$OUT/serve-stress-verdict.txt"
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert len(rows) == 64
assert all(row.get("ok") is True for row in rows)
assert all(row.get("done") is True for row in rows)
assert all(row.get("finish_reason") in ("stop", "length") for row in rows)
print("serve_stress_c64=PASS requests=64/64 well_formed=64/64")
PY
wait_idle

grep -Ein \
    'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' \
    "$OUT/serve-smoke-q35-server.log" "$OUT/serve-stress-server.log" \
    >"$OUT/server-failure-scan.log" || true
test ! -s "$OUT/server-failure-scan.log" || {
    cat "$OUT/server-failure-scan.log"
    echo "FAIL: server failure signature captured"
    exit 1
}

stop_sampler
python3 - "$OUT/gpu-1s.csv" <<'PY' | tee "$OUT/thermal-summary.txt"
import csv
import sys

rows = list(csv.reader(open(sys.argv[1], encoding="utf-8")))
for gpu in (0, 1):
    selected = [row for row in rows if len(row) >= 11 and int(row[1].strip()) == gpu]
    assert selected, f"no thermal samples for GPU {gpu}"
    temp = [float(row[4]) for row in selected]
    clock = [float(row[5]) for row in selected]
    power = [float(row[6]) for row in selected]
    memory = [float(row[8]) for row in selected]
    util = [float(row[10]) for row in selected]
    print(
        f"gpu={gpu} samples={len(selected)} temp_C={min(temp):.0f}-{max(temp):.0f} "
        f"clock_MHz={min(clock):.0f}-{max(clock):.0f} max_power_W={max(power):.2f} "
        f"max_memory_MiB={max(memory):.0f} max_util_pct={max(util):.0f}"
    )
PY

{
    echo "battery_runs_n=1"
    echo "kernel_check_n=1"
    echo "run_gen_q27_n=1"
    echo "run_gen_q35_n=1"
    echo "run_spec_q27_k_sweep=1..8, one run per K"
    echo "run_spec_q35_k_sweep=1..8, one run per K"
    echo "serve_smoke_n=1"
    echo "q35_mixed_c4_requests_n=20"
    echo "q35_mixed_c4_completion_tokens_each=60"
    echo "serve_stress_c64_requests_n=64"
} | tee "$OUT/run-shapes.txt"

snapshot "$OUT/gpu-after.log" after
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained after battery"; exit 77; }
stop_owned_servers

remaining=$(pgrep -af 'target/release/(memra-server|kernel-check|run-gen|run-spec)' || true)
listeners=$(ss -H -ltnp 2>/dev/null | grep -E ':(8177|8179|18427|18435)\b' || true)
{
    echo "pre_unlock_ts=$(date -u +%FT%TZ)"
    echo "remaining_runtime_processes=${remaining:-none}"
    echo "gate_port_listeners=${listeners:-none}"
} | tee "$OUT/pre-unlock-clean.log"
test -z "$remaining"
test -z "$listeners"

echo "COLDSHIP_LOCK_RELEASING ts=$(date -u +%FT%TZ)"
flock -u 9
exec 9>&-
lock_held=0

remaining=$(pgrep -af 'target/release/(memra-server|kernel-check|run-gen|run-spec)' || true)
listeners=$(ss -H -ltnp 2>/dev/null | grep -E ':(8177|8179|18427|18435)\b' || true)
lock_probe=FAIL
flock -n /tmp/memra-gpu.lock true && lock_probe=PASS
{
    echo "post_unlock_ts=$(date -u +%FT%TZ)"
    echo "remaining_runtime_processes=${remaining:-none}"
    echo "gate_port_listeners=${listeners:-none}"
    echo "lock_reacquire_probe=$lock_probe"
} | tee "$OUT/post-unlock-clean.log"
test -z "$remaining"
test -z "$listeners"
test "$lock_probe" = PASS

touch "$OUT/battery.ok"
echo "COLDSHIP_BATTERY_ALL_GREEN ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
trap - EXIT INT TERM
