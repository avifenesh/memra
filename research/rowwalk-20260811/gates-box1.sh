#!/usr/bin/env bash
# One bounded rowwalk correctness or golden block on box1.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  correctness|golden) ;;
  *) echo "usage: $0 correctness|golden" >&2; exit 2 ;;
esac

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged candidate commit}"
REPO=${REPO:-$HOME/memra-cx-rowwalk}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
GOLDEN=${GOLDEN:-$HOME/darktrain2/golden-response.bin}
EXPECTED_GOLDEN=${EXPECTED_GOLDEN:-21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de}
PORT=${PORT:-18436}
BASE=http://127.0.0.1:$PORT
STAMP=${ROWWALK_GATES_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/rowwalk/gates/$STAMP/$MODE}

SERVER=$REPO/target/release/memra-server
KERNEL=$REPO/target/release/kernel-check
BATCH=$REPO/target/release/decode-batch-gate
GEN=$REPO/target/release/run-gen
SPEC=$REPO/target/release/run-spec
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PROMPT=$REPO/tools/fast-gate/prompts/probe.txt
SERVER_PID=0

if [[ $MODE == correctness ]]; then
  : "${EXPECTED_KERNEL:?set EXPECTED_KERNEL to the release binary SHA-256}"
  : "${EXPECTED_BATCH:?set EXPECTED_BATCH to the release binary SHA-256}"
  : "${EXPECTED_GEN:?set EXPECTED_GEN to the release binary SHA-256}"
  : "${EXPECTED_SPEC:?set EXPECTED_SPEC to the release binary SHA-256}"
else
  : "${EXPECTED_SERVER:?set EXPECTED_SERVER to the release binary SHA-256}"
fi

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
  local path=$1 label=$2
  {
    echo "label=$label"
    echo "ts=$(date -u +%FT%TZ)"
    nvidia-smi \
      --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
      --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader
  } > "$path" 2>&1
}

cleanup() {
  if (( SERVER_PID > 0 )); then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
}

wait_idle() {
  local _
  for _ in $(seq 1 120); do
    [[ -z $(compute_apps) ]] && return 0
    sleep 1
  done
  compute_apps
  return 1
}

wait_ready() {
  local log=$1 _
  for _ in $(seq 1 900); do
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -120 "$log"
      return 1
    fi
    sleep 1
  done
  tail -120 "$log"
  return 1
}

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  echo "binary=$path sha256=$actual"
  [[ $actual == "$expected" ]]
}

run_logged() {
  local label=$1 timeout_s=$2
  shift 2
  echo "gate_start=$label ts=$(date -u +%FT%TZ)"
  set +e
  timeout "$timeout_s" "$@" 2>&1 | tee "$OUT/$label.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$rc" > "$OUT/$label.exit"
  echo "gate_done=$label ts=$(date -u +%FT%TZ) rc=$rc"
  return "$rc"
}

preflight() {
  local source apps
  source=$(git -C "$REPO" rev-parse HEAD)
  echo "source=$source"
  [[ $source == "$EXPECTED_SOURCE" ]]
  git -C "$REPO" status --short --branch --untracked-files=no
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
  if [[ $MODE == correctness ]]; then
    check_hash "$KERNEL" "$EXPECTED_KERNEL"
    check_hash "$BATCH" "$EXPECTED_BATCH"
    check_hash "$GEN" "$EXPECTED_GEN"
    check_hash "$SPEC" "$EXPECTED_SPEC"
  else
    check_hash "$SERVER" "$EXPECTED_SERVER"
    [[ -f $QOS && -f $GOLDEN ]]
    check_hash "$GOLDEN" "$EXPECTED_GOLDEN"
    echo "qos_sha256=$(sha256sum "$QOS" | awk '{print $1}')"
  fi
  apps=$(compute_apps)
  [[ -z $apps ]] || { echo "$apps"; return 1; }
}

run_correctness() {
  run_logged kernel-check 3600 env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
    "$KERNEL" "$MODEL"
  grep -q 'ALL GREEN: kernels match CPU reference' "$OUT/kernel-check.log"
  grep -q 'fa_decode packed-row views vs copied rows hd=128 B=4: bitdiff=0 OK' \
    "$OUT/kernel-check.log"

  run_logged decode-batch-gate 7200 env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    "$BATCH" "$MODEL" --mode pp --batch 1,2,4,8 --steps 24 --reps 2 \
    --stages 2 --plen 520
  grep -q 'pp mode verdict: 0 failing arm(s)' "$OUT/decode-batch-gate.log"
  grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' \
    "$OUT/decode-batch-gate.log"

  run_logged run-gen 3600 env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_NGEN=64 \
    MEMRA_PROMPT_FILE="$PROMPT" "$GEN" "$MODEL"
  [[ $(grep -c 'MATCH' "$OUT/run-gen.log") -ge 2 ]]

  run_logged run-spec 7200 env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_NGEN=32 \
    MEMRA_MTP_DRAFT="$DRAFT" MEMRA_PROMPT_FILE="$PROMPT" "$SPEC" "$MODEL"
  [[ $(grep -c 'self-consistency: PASS' "$OUT/run-spec.log") -eq 8 ]]
  grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"
}

run_golden() {
  local log=$OUT/server.log
  env -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
    -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" MEMRA_COMPAT=openai \
    MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_PREFIX_CACHE_MB=256 "$SERVER" > "$log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$log"
  python3 "$QOS" --base "$BASE" --model "$MODEL_NAME" --label rowwalk-golden \
    --requests 1 --max-tokens 64 --golden "$GOLDEN" \
    --rows "$OUT/qos-rows.jsonl" --summary "$OUT/qos-summary.json"
  python3 - "$OUT/qos-summary.json" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["exactness"] == "match", row
assert row["n_ok"] == 1 and row["n_error"] == 0, row
assert row["golden_matches"] == 1 and row["golden_divergences"] == 0, row
print("golden_receipt=PASS")
PY
  python3 - "$OUT/qos-summary.json" "$OUT/golden-receipt.json" \
    "$EXPECTED_SOURCE" "$EXPECTED_SERVER" "$EXPECTED_GOLDEN" <<'PY'
import json
import pathlib
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
receipt = {
    "schema": "memra.rowwalk.golden-receipt.v1",
    "candidate_source": sys.argv[3],
    "candidate_binary_sha256": sys.argv[4],
    "golden_sha256": sys.argv[5],
    "summary": summary,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  cleanup
  wait_idle
  if grep -Ein 'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal' "$log"; then
    return 1
  fi
}

(
  flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
  trap cleanup EXIT INT TERM
  echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) mode=$MODE stamp=$STAMP"
  preflight
  snapshot "$OUT/nvidia-smi-before.log" preflight
  if [[ $MODE == correctness ]]; then run_correctness; else run_golden; fi
  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ) result=PASS"
) 9>/tmp/memra-gpu.lock
