#!/usr/bin/env bash
# Five-round whole-server interleaved before/after binary pair under one lock hold.
set -euo pipefail

ROOT=${COLDHOL_ROOT:-/opt/dl-image/nvme/cx-coldhol}
REPO=${COLDHOL_REPO:-$ROOT/memra}
RUNNER=${COLDHOL_RUNNER:-$REPO/research/coldhol-20260812/run-box1.sh}
EXPECTED_HARNESS_SOURCE=${COLDHOL_EXPECTED_HARNESS_SOURCE:?set the checked-out harness commit}
BASE_SERVER=${COLDHOL_BASE_SERVER:-$ROOT/binaries/base/memra-server}
CANDIDATE_SERVER=${COLDHOL_CANDIDATE_SERVER:-$ROOT/binaries/candidate/memra-server}
BASE_SHA256=${COLDHOL_BASE_SHA256:-b5e31c8db47f2d5f04a2ffb8729c921fd4b68cb6f090819b8234eb0996385ef3}
CANDIDATE_SHA256=${COLDHOL_CANDIDATE_SHA256:-f00f1bd5d08fbf0476a540e497b51d749d813873c4b885a67fc5fce120120748}
BASE_SOURCE=${COLDHOL_BASE_SOURCE:-d2fba620031920032b253b700443af5ef1ec7866}
CANDIDATE_SOURCE=${COLDHOL_CANDIDATE_SOURCE:-b37d77c6f6403d8b3b87099470fc3b5c2cd62cee}
LEVELS=${COLDHOL_LEVELS:-8,12,16,20,24}
ROUNDS=${COLDHOL_AB_ROUNDS:-5}
STAMP=${COLDHOL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${COLDHOL_OUT:-$ROOT/raw/ab-coldchunks-$STAMP}

test "$LEVELS" = 8,12,16,20,24 || { echo "FAIL: frozen levels changed" >&2; exit 2; }
test "$ROUNDS" -eq 5 || { echo "FAIL: promotion A/B requires exactly five rounds" >&2; exit 2; }
test -x "$RUNNER"
test ! -e "$OUT" || { echo "FAIL: output exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/orchestrator.log") 2>&1

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "COLDHOL_AB_FAIL ts=$(date -u +%FT%TZ) rc=$rc out=$OUT"
    fi
}
trap cleanup EXIT INT TERM

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "COLDHOL_AB_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"

{
    echo "label=before-ab"
    echo "ts=$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,\
memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
} >"$OUT/gpu-before.log"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 GPUs are not idle"; exit 1; }
sha256sum "$BASE_SERVER" "$CANDIDATE_SERVER" >"$OUT/binary-pair.sha256"
test "$(sha256sum "$BASE_SERVER" | awk '{print $1}')" = "$BASE_SHA256"
test "$(sha256sum "$CANDIDATE_SERVER" | awk '{print $1}')" = "$CANDIDATE_SHA256"

run_arm() {
    local round=$1 order=$2 arm=$3
    local label="ab-coldchunks-$arm-r$round"
    local child
    child="$OUT/r$(printf '%02d' "$round")-o$(printf '%02d' "$order")-$arm"
    local server sha source expect_partial
    if [[ $arm == baseline ]]; then
        server=$BASE_SERVER
        sha=$BASE_SHA256
        source=$BASE_SOURCE
        expect_partial=no
    else
        server=$CANDIDATE_SERVER
        sha=$CANDIDATE_SHA256
        source=$CANDIDATE_SOURCE
        expect_partial=yes
    fi
    echo "AB_ARM_START ts=$(date -u +%FT%TZ) round=$round order=$order arm=$arm"
    COLDHOL_LOCK_HELD=1 \
    COLDHOL_EXPECTED_HARNESS_SOURCE="$EXPECTED_HARNESS_SOURCE" \
    COLDHOL_SERVER="$server" \
    COLDHOL_EXPECTED_SERVER_SHA256="$sha" \
    COLDHOL_RUNTIME_SOURCE="$source" \
    COLDHOL_EXPECT_PARTIAL="$expect_partial" \
    COLDHOL_PRIME_BATCH='' \
    COLDHOL_LABEL="$label" \
    COLDHOL_OUT="$child" \
    COLDHOL_REPS=1 \
    COLDHOL_REP_START="$round" \
    COLDHOL_LEVELS="$LEVELS" \
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

{
    echo "label=after-ab"
    echo "ts=$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,\
memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
} >"$OUT/gpu-after.log"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained"; exit 1; }

find "$OUT" -mindepth 2 -maxdepth 2 -name run.ok -print | sort >"$OUT/run-ok-files.txt"
test "$(wc -l <"$OUT/run-ok-files.txt")" -eq $((ROUNDS * 2))
find "$OUT" -type f ! -name MANIFEST.sha256 ! -name orchestrator.log -print0 \
    | sort -z | xargs -0 sha256sum >"$OUT/MANIFEST.sha256"
touch "$OUT/ab.ok"
echo "COLDHOL_AB_PASS ts=$(date -u +%FT%TZ) out=$OUT"
trap - EXIT INT TERM
