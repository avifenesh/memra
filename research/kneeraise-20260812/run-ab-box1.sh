#!/usr/bin/env bash
# Five-round, whole-server interleaved A/B wrapper for one Q27 serving knob.
set -euo pipefail

ROOT=${KNEERAISE_ROOT:-/opt/scratch/nvme/cx-kneeraise}
HARNESS=${KNEERAISE_HARNESS:-$ROOT/harness}
RUNNER=${KNEERAISE_RUNNER:-$HARNESS/run-box1.sh}
NAME=${KNEERAISE_AB_NAME:-prefill2048}
CANDIDATE_PREFILL_TICK=${KNEERAISE_CANDIDATE_PREFILL_TICK:-}
CANDIDATE_DECODE_CAP=${KNEERAISE_CANDIDATE_DECODE_CAP:-}
LEVELS=${KNEERAISE_LEVELS:-8,12,16,20,24}
ROUNDS=${KNEERAISE_AB_ROUNDS:-5}
STAMP=${KNEERAISE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${KNEERAISE_OUT:-$ROOT/raw/ab-$NAME-$STAMP}

changes=0
test -z "$CANDIDATE_PREFILL_TICK" || changes=$((changes + 1))
test -z "$CANDIDATE_DECODE_CAP" || changes=$((changes + 1))
test "$changes" -eq 1 || {
    echo "FAIL: define exactly one candidate serving knob" >&2
    exit 2
}
test "$ROUNDS" -eq 5 || {
    echo "FAIL: promotion-grade A/B requires exactly five rounds" >&2
    exit 2
}
test -x "$RUNNER"
test ! -e "$OUT" || { echo "FAIL: output exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/orchestrator.log") 2>&1

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "KNEERAISE_AB_FAIL ts=$(date -u +%FT%TZ) rc=$rc out=$OUT"
    fi
}
trap cleanup EXIT INT TERM

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "KNEERAISE_AB_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"

nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader >"$OUT/gpu-before.log"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 GPUs are not idle"; exit 1; }

run_arm() {
    local round=$1 order=$2 arm=$3
    local label="ab-$NAME-$arm-r$round"
    local child="$OUT/r$(printf '%02d' "$round")-o$(printf '%02d' "$order")-$arm"
    local prefill_tick= decode_cap=
    if [[ $arm == candidate ]]; then
        prefill_tick=$CANDIDATE_PREFILL_TICK
        decode_cap=$CANDIDATE_DECODE_CAP
    fi
    echo "AB_ARM_START ts=$(date -u +%FT%TZ) round=$round order=$order arm=$arm"
    KNEERAISE_LOCK_HELD=1 \
    KNEERAISE_LABEL="$label" \
    KNEERAISE_OUT="$child" \
    KNEERAISE_REPS=1 \
    KNEERAISE_REP_START="$round" \
    KNEERAISE_LEVELS="$LEVELS" \
    KNEERAISE_PREFIX_CACHE_MB=4096 \
    KNEERAISE_MAX_SESSIONS=96 \
    KNEERAISE_PREFILL_TICK="$prefill_tick" \
    KNEERAISE_DECODE_BATCH_CAP="$decode_cap" \
        "$RUNNER" run
    test -e "$child/run.ok"
    echo "AB_ARM_PASS ts=$(date -u +%FT%TZ) round=$round order=$order arm=$arm"
}

for round in $(seq 1 "$ROUNDS"); do
    if (( round % 2 == 1 )); then
        arms=(baseline candidate)
    else
        arms=(candidate baseline)
    fi
    order=0
    for arm in "${arms[@]}"; do
        order=$((order + 1))
        run_arm "$round" "$order" "$arm"
    done
done

nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader >"$OUT/gpu-after.log"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained"; exit 1; }

find "$OUT" -mindepth 2 -maxdepth 2 -name run.ok -print | sort >"$OUT/run-ok-files.txt"
test "$(wc -l <"$OUT/run-ok-files.txt")" -eq $((ROUNDS * 2))
find "$OUT" -type f ! -name MANIFEST.sha256 ! -name orchestrator.log -print0 \
    | sort -z | xargs -0 sha256sum >"$OUT/MANIFEST.sha256"
touch "$OUT/ab.ok"
echo "KNEERAISE_AB_PASS ts=$(date -u +%FT%TZ) out=$OUT"
trap - EXIT INT TERM
