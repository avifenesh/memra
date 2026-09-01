#!/usr/bin/env bash
# Same-window acceptance settle for the local-ci 26b-spec-d1736 red cell.
set -euo pipefail

REPO=${SIGROUTER2_REPO:-/home/avifenesh/projects/wt-cx-sigrouter2}
BASE=${1:?usage: local-26b-ab.sh BASELINE_GEMMA_GATE CANDIDATE_GEMMA_GATE}
CAND=${2:?usage: local-26b-ab.sh BASELINE_GEMMA_GATE CANDIDATE_GEMMA_GATE}
OUT=${SIGROUTER2_26B_OUT:-$REPO/research/sigrouter2-20260811/raw/local-26b-ab}
MODEL=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
DRAFT=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf
PROMPT=$REPO/research/gemma4-bringup/depth-prompt-1736-ids.txt
RANKS=$REPO/research/gemma4-bringup/gemma4-26b-owngen-ranks-32768.gguf.txt
N=5

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
for artifact in "$BASE" "$CAND" "$MODEL" "$DRAFT" "$PROMPT" "$RANKS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=name,uuid,temperature.gpu,pstate,clocks.sm,power.draw,power.limit,memory.used \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
            --format=csv,noheader,nounits
    } >"$path" 2>&1
}

run_one() {
    local arm=$1 rep=$2 bin=$3
    local log="$OUT/r${rep}-${arm}.log"
    echo "arm=$arm rep=$rep start=$(date -u +%FT%TZ)"
    # The prompt file is an argv list of integer token ids.
    # shellcheck disable=SC2046
    env \
        MEMRA_SPEC_ONLY=1 \
        MEMRA_SPEC=6 \
        MEMRA_DRAFT="$DRAFT" \
        MEMRA_NGEN=128 \
        MEMRA_GEMMA_DRAFT_RANKS="$RANKS" \
        timeout 420 "$bin" "$MODEL" $(tr '\n' ' ' < "$PROMPT") >"$log" 2>&1
    local toks accept tok_round rounds drafted accepted
    toks=$(grep -oE 'spec: [0-9.]+' "$log" | grep -oE '[0-9.]+' | tail -1)
    accept=$(grep -oE 'accept-rate=[0-9.]+' "$log" | grep -oE '[0-9.]+' | tail -1)
    tok_round=$(grep -oE 'tok/round=[0-9.]+' "$log" | grep -oE '[0-9.]+' | tail -1)
    rounds=$(grep -oE 'rounds=[0-9]+' "$log" | grep -oE '[0-9]+' | tail -1)
    drafted=$(grep -oE 'drafted=[0-9]+' "$log" | grep -oE '[0-9]+' | tail -1)
    accepted=$(grep -oE 'accepted=[0-9]+' "$log" | grep -oE '[0-9]+' | tail -1)
    test -n "$toks" && test -n "$accept" && test -n "$tok_round"
    printf '%s\n' "$toks" >>"$OUT/$arm.toks"
    printf '%s\n' "$accept" >>"$OUT/$arm.accept"
    jq -cn \
        --arg arm "$arm" --argjson rep "$rep" --argjson toks "$toks" \
        --argjson accept "$accept" --argjson tok_round "$tok_round" \
        --argjson rounds "$rounds" --argjson drafted "$drafted" --argjson accepted "$accepted" \
        '{arm:$arm,rep:$rep,tok_s:$toks,accept:$accept,tok_round:$tok_round,rounds:$rounds,drafted:$drafted,accepted:$accepted}' \
        >>"$OUT/points.jsonl"
    local thermal
    thermal=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')
    echo "arm=$arm rep=$rep tok_s=$toks accept=$accept tok_round=$tok_round rounds=$rounds drafted=$drafted accepted=$accepted thermal=$thermal"
}

median() {
    sort -g "$1" | awk '{v[NR]=$1} END {if (NR != 5) exit 1; print v[3]}'
}

cd "$REPO"
exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "AB_LOCK_ACQUIRED $(date -u +%FT%TZ)"
apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU not idle after lock acquisition"; exit 1; }
echo "candidate_commit=$(git rev-parse HEAD) baseline_commit=30418923a2bf17eef3fe989bfc5f1d5f50db0db4"
sha256sum "$BASE" "$CAND" "$MODEL" "$DRAFT" "$PROMPT" "$RANKS" >"$OUT/SHA256SUMS"
snapshot "$OUT/thermal-before.log" before

# One discarded candidate warmup brings the shared rig into the measured thermal regime.
# The prompt file is an argv list of integer token ids.
# shellcheck disable=SC2046
env MEMRA_SPEC_ONLY=1 MEMRA_SPEC=6 MEMRA_DRAFT="$DRAFT" MEMRA_NGEN=32 \
    MEMRA_GEMMA_DRAFT_RANKS="$RANKS" timeout 420 "$CAND" "$MODEL" \
    $(tr '\n' ' ' < "$PROMPT") >"$OUT/warmup.log" 2>&1

for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        run_one baseline "$rep" "$BASE"
        run_one candidate "$rep" "$CAND"
    else
        run_one candidate "$rep" "$CAND"
        run_one baseline "$rep" "$BASE"
    fi
done

snapshot "$OUT/thermal-after.log" after
base_median=$(median "$OUT/baseline.toks")
cand_median=$(median "$OUT/candidate.toks")
base_accept=$(sort -u "$OUT/baseline.accept")
cand_accept=$(sort -u "$OUT/candidate.accept")
test "$(wc -l < "$OUT/baseline.accept")" -eq "$N"
test "$(wc -l < "$OUT/candidate.accept")" -eq "$N"
test "$(printf '%s\n' "$base_accept" | wc -l)" -eq 1
test "$(printf '%s\n' "$cand_accept" | wc -l)" -eq 1
test "$base_accept" = "$cand_accept" || {
    echo "FAIL: acceptance differs baseline=$base_accept candidate=$cand_accept"
    exit 1
}
delta=$(awk -v base="$base_median" -v cand="$cand_median" \
    'BEGIN { printf "%.4f", 100.0 * (cand / base - 1.0) }')
jq -s \
    --argjson baseline_median "$base_median" --argjson candidate_median "$cand_median" \
    --argjson delta_pct "$delta" --argjson acceptance "$base_accept" \
    '{schema:"memra.sigrouter2.26b-ab.v1",runs_per_arm:5,baseline_commit:"30418923a2bf17eef3fe989bfc5f1d5f50db0db4",baseline_median_tok_s:$baseline_median,candidate_median_tok_s:$candidate_median,candidate_delta_pct:$delta_pct,acceptance_both_arms:$acceptance,points:.}' \
    "$OUT/points.jsonl" >"$OUT/summary.json"
cat "$OUT/summary.json"
echo "AB_PASS $(date -u +%FT%TZ)"
