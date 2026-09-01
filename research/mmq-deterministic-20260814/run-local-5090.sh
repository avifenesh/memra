#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-"$ROOT/research/mmq-deterministic-20260814/raw/$STAMP"}
MODEL=${MODEL:-/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf}
DRAFT=${DRAFT:-/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf}
RANKS_SRC=${RANKS_SRC:-"$ROOT/research/gemma4-bringup/gemma4-26b-owngen-ranks-32768.gguf.txt"}
PROMPT_IDS=${PROMPT_IDS:-"$ROOT/research/gemma4-bringup/depth-prompt-1736-ids.txt"}
BINARY=${BINARY:-"$ROOT/target/release/gemma-gate"}
LOCK=${LOCK:-/tmp/memra-5090.lock}

test ! -e "$OUT" || {
  echo "refusing to overwrite $OUT" >&2
  exit 1
}
mkdir -p "$OUT"
for path in "$MODEL" "$DRAFT" "$RANKS_SRC" "$PROMPT_IDS" "$BINARY"; do
  test -f "$path" || {
    echo "missing required file: $path" >&2
    exit 1
  }
done

exec > >(tee "$OUT/driver.log") 2>&1
echo "CAMPAIGN=mmq-deterministic-tile-vs-sk"
echo "START_UTC=$(date -u +%FT%TZ)"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "LOCK=$LOCK"

exec 9>"$LOCK"
flock -n 9 || {
  echo "canonical lock busy: $LOCK" >&2
  exit 75
}
echo "LOCK_ACQUIRED=1"

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

test -z "$(compute_apps)" || {
  echo "unexpected compute applications before campaign" >&2
  compute_apps
  exit 1
}

nvidia-smi \
  --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 >"$OUT/telemetry-250ms.csv" 2>&1 &
SAMPLER_PID=$!
cleanup() {
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

sha256sum "$BINARY" "$MODEL" "$DRAFT" "$RANKS_SRC" "$PROMPT_IDS" >"$OUT/hashes.sha256"
git -C "$ROOT" rev-parse HEAD >"$OUT/base-commit.txt"
git -C "$ROOT" diff --binary >"$OUT/candidate.diff"
sha256sum "$OUT/candidate.diff" >>"$OUT/hashes.sha256"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"

IFS=' ' read -r -a prompt_ids <"$PROMPT_IDS" || test "${#prompt_ids[@]}" -gt 0
test "${#prompt_ids[@]}" -gt 0
echo "PROMPT_TOKENS=${#prompt_ids[@]}"

seqno=0
run_arm() {
  local label=$1
  shift
  seqno=$((seqno + 1))
  local prefix
  prefix=$(printf "%02d-%s" "$seqno" "$label")
  local arm_tmp
  arm_tmp=$(mktemp -d "/tmp/memra-mmq-${label}.XXXXXX")
  cp "$RANKS_SRC" "$arm_tmp/ranks.txt"
  echo "ARM=$label SEQ=$seqno START_UTC=$(date -u +%FT%TZ)"
  echo "ARM=$label COMPUTE_APPS_BEFORE"
  compute_apps
  set +e
  env \
    -u MEMRA_SPEC_ONLY \
    -u MEMRA_MMQ_SK \
    -u MEMRA_MMQ_SK_FORM \
    -u MEMRA_MMQ_SK_DEBUG \
    MEMRA_SPEC=6 \
    MEMRA_DRAFT="$DRAFT" \
    MEMRA_NGEN=128 \
    MEMRA_GATE_DUMP_TOKENS=1 \
    MEMRA_GEMMA_DRAFT_RANKS="$arm_tmp/ranks.txt" \
    "$@" timeout 420 "$BINARY" "$MODEL" "${prompt_ids[@]}" \
    2>&1 | tee "$OUT/$prefix.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "ARM=$label SEQ=$seqno RC=$rc END_UTC=$(date -u +%FT%TZ)"
  rm -rf "$arm_tmp"
  test "$rc" -eq 0
}

for rep in 1 2 3 4 5; do
  if (( rep % 2 == 1 )); then
    run_arm "r${rep}-tile" MEMRA_MMQ_SK=1 MEMRA_MMQ_SK_FORM=tile MEMRA_MMQ_SK_DEBUG=1
    run_arm "r${rep}-sk" MEMRA_MMQ_SK=1 MEMRA_MMQ_SK_FORM=sk MEMRA_MMQ_SK_DEBUG=1
  else
    run_arm "r${rep}-sk" MEMRA_MMQ_SK=1 MEMRA_MMQ_SK_FORM=sk MEMRA_MMQ_SK_DEBUG=1
    run_arm "r${rep}-tile" MEMRA_MMQ_SK=1 MEMRA_MMQ_SK_FORM=tile MEMRA_MMQ_SK_DEBUG=1
  fi
done

run_arm naked-debug MEMRA_MMQ_SK_DEBUG=1
run_arm naked-clean

cleanup
trap - EXIT
echo "POST_COMPUTE_APPS"
compute_apps
test -z "$(compute_apps)" || {
  echo "compute applications remain after campaign" >&2
  exit 1
}
"$ROOT/research/mmq-deterministic-20260814/summarize.py" "$OUT"
sha256sum "$OUT"/*.log "$OUT/telemetry-250ms.csv" >>"$OUT/hashes.sha256"
echo "CAMPAIGN_PASS=1"
echo "END_UTC=$(date -u +%FT%TZ)"
