#!/usr/bin/env bash
# One exclusive box1 lock hold: clean release build, N=5 MoESD sweep, then exactness battery.
set -euo pipefail

REPO=${MOESD_REPO:-/home/ubuntu/memra-cx-moesd}
OUT=${MOESD_OUT:-/home/ubuntu/moesd-receipts/box1}
MODEL_ROOT=${MOESD_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${MOESD_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${MOESD_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_DIR=$(dirname "$MODEL")
BIN=$REPO/target/release
LANE=$REPO/research/moesd-harness-20260811
EXPECTED_SOURCE=${MOESD_EXPECTED_SOURCE:?'FAIL: MOESD_EXPECTED_SOURCE is required (use the exact 40-character commit, or explicit any opt-out)'}
CARGO=${CARGO:-/home/ubuntu/.cargo/bin/cargo}
RUSTC=${RUSTC:-/home/ubuntu/.cargo/bin/rustc}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

sampler_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
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

stop_sampler() {
    local pid=${sampler_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sampler_pid=
}

wait_zero() {
    local apps memory
    for _ in $(seq 1 120); do
        apps=$(compute_apps)
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
        if [[ -z $apps && $memory == "0 0 " ]]; then
            return 0
        fi
        sleep 1
    done
    compute_apps
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    return 1
}

run_capture() {
    local name=$1
    shift
    echo "=== $name $(date -u +%FT%TZ) ==="
    "$@" > >(tee "$OUT/${name}.log") 2>&1
    echo "PASS: $name rc=0 $(date -u +%FT%TZ)"
}

trap stop_sampler EXIT INT TERM

mapfile -d '' MODEL_PARTS < <(
    find "$MODEL_DIR" -maxdepth 1 -type f -name 'Step-3.7-flash-IQ4_XS-*.gguf' -print0 | sort -z
)
test "${#MODEL_PARTS[@]}" -eq 3 || {
    echo "FAIL: expected three IQ4_XS model shards under $MODEL_DIR, found ${#MODEL_PARTS[@]}"
    exit 1
}
for artifact in "${MODEL_PARTS[@]}" "$DRAFT" "$LANE/sample-nvml.py" "$LANE/summarize.py"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
test -x "$CARGO" || { echo "FAIL: cargo is not executable: $CARGO"; exit 1; }
test -x "$RUSTC" || { echo "FAIL: rustc is not executable: $RUSTC"; exit 1; }

exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo "FAIL: box1 GPU lock is occupied; refusing to contend"; exit 75; }
echo "MOESD_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
test -z "$(git status --porcelain)" || {
    git status --short --branch
    echo "FAIL: remote study worktree is dirty"
    exit 1
}
if [[ $EXPECTED_SOURCE != any && $SOURCE_COMMIT != "$EXPECTED_SOURCE" ]]; then
    echo "FAIL: source commit $SOURCE_COMMIT != expected $EXPECTED_SOURCE"
    exit 1
fi
git status --short --branch
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

echo "RELEASE_BUILD_START $(date -u +%FT%TZ)"
nice -n 15 taskset -c 0-15 "$CARGO" build --release -p memra-engine \
    --bin target-efficiency --bin kernel-check --bin run-gen --bin run-spec \
    --bin decode-batch-gate >"$OUT/build.log" 2>&1
echo "RELEASE_BUILD_PASS $(date -u +%FT%TZ)"

sha256sum "${MODEL_PARTS[@]}" "$DRAFT" "$BIN/target-efficiency" "$BIN/kernel-check" \
    "$BIN/run-gen" "$BIN/run-spec" "$BIN/decode-batch-gate" \
    "$LANE/run-box1.sh" "$LANE/sample-nvml.py" "$LANE/summarize.py" >"$OUT/SHA256SUMS"
{
    "$RUSTC" --version
    "$CARGO" --version
    nvcc --version
    uname -a
    git rev-parse HEAD
} >"$OUT/manifest.txt" 2>&1

python3 "$LANE/sample-nvml.py" "$OUT/thermal.jsonl" >"$OUT/sampler.log" 2>&1 &
sampler_pid=$!
env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    MEMRA_MTP_DRAFT="$DRAFT" \
    timeout 7200 "$BIN/target-efficiency" "$MODEL" \
        --b 1,2,4,8,16,24,32 --gamma 1,2,3,4,6,8 --runs 5 \
        >"$OUT/measurements.jsonl" 2>"$OUT/harness.log"
stop_sampler
python3 "$LANE/summarize.py" \
    "$OUT/measurements.jsonl" "$OUT/thermal.jsonl" "$OUT/raw.jsonl" \
    "$OUT/RESULTS.jsonl" "$OUT/summary.json" | tee "$OUT/summarize.log"

run_capture kernel-check timeout 3600 env CUDA_VISIBLE_DEVICES=0 \
    "$BIN/kernel-check" "$MODEL"
grep -Eq '^ALL GREEN \([0-9]+ cells, [0-9]+ skipped\)$' "$OUT/kernel-check.log"
test "$(grep -c ' OK' "$OUT/kernel-check.log")" -gt 300

run_capture run-gen timeout 3600 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_NGEN=64 \
    "$BIN/run-gen" "$MODEL" --prompt \
    'Explain in one short paragraph why batching can amortize memory-bound expert weights.'
test "$(grep -c ' MATCH' "$OUT/run-gen.log")" -eq 2
! grep -q 'MISMATCH' "$OUT/run-gen.log"

run_capture run-spec timeout 3600 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_MTP_DRAFT="$DRAFT" \
    MEMRA_NGEN=32 \
    "$BIN/run-spec" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"

run_capture decode-batch-pp timeout 3600 env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    "$BIN/decode-batch-gate" "$MODEL" --mode pp --stages 2 \
    --batch 1,2,4,8 --steps 16 --reps 2 --plen 520
grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' "$OUT/decode-batch-pp.log"

snapshot "$OUT/nvidia-smi-after.log" complete
wait_zero
echo "MOESD_PASS $(date -u +%FT%TZ)"
