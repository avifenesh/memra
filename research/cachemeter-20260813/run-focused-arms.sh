#!/usr/bin/env bash
# Reproduce only serve-smoke's cache-metering and Q35 mixed-c=4 arms without
# changing either gate. The final lane still runs tools/serve-smoke.sh through
# tools/local-ci.sh --perf-quick.
set -uo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 LABEL REPO_ROOT OUTPUT_DIR" >&2
  exit 2
fi

LABEL=$1
REPO_ROOT=$(realpath "$2")
OUT_DIR=$(realpath -m "$3")
TMPDIR=${TMPDIR:-/home/avifenesh/tmp-lanes}
export TMPDIR

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q35_MODEL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ADDR=127.0.0.1:8177
BASE=http://$ADDR
SPID=

mkdir -p "$OUT_DIR"

stop_server() {
  if [ -n "${SPID:-}" ]; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=
  fi
}

start_server() {
  local models=$1 log=$2
  MEMRA_COMPAT=openai MEMRA_MODELS="$models" MEMRA_ADDR=$ADDR \
    "$REPO_ROOT/target/release/memra-server" >"$log" 2>&1 &
  SPID=$!
  for _ in $(seq 120); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "server did not come up; log tail:"
  tail -5 "$log"
  return 1
}

trap stop_server EXIT

echo "label=$LABEL"
echo "repo_root=$REPO_ROOT"
echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "tree=$(git -C "$REPO_ROOT" rev-parse 'HEAD^{tree}')"
echo "prefix_policy=${MEMRA_PREFIX_CACHE_POLICY-<unset>}"
echo "tmpdir=$TMPDIR"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true

if [ ! -f "$MODEL" ] || [ ! -f "$Q35_MODEL" ]; then
  echo "focused-arms: required model missing"
  exit 2
fi

(cd "$REPO_ROOT" && cargo build --release -p memra-server) 2>&1 | tee "$OUT_DIR/build.log"
build_status=${PIPESTATUS[0]}
if [ "$build_status" -ne 0 ]; then
  echo "focused-arms: build_status=$build_status"
  exit "$build_status"
fi
sha256sum "$REPO_ROOT/target/release/memra-server" | tee "$OUT_DIR/binary.sha256"

echo "== focused serve-smoke arm: cache-metering exactness =="
export MEMRA_SERVE_SPEC=0
if start_server "smoke=$MODEL" "$OUT_DIR/cache-meter-server.log"; then
  (cd "$REPO_ROOT" && python3 tools/cache-meter-gate.py "$BASE" smoke \
    --n 5 --k 256 --suffix 16 \
    --raw-out "$OUT_DIR/cache-meter-raw.jsonl") \
    2>&1 | tee "$OUT_DIR/cache-meter-gate.log"
  cache_status=${PIPESTATUS[0]}
  stop_server
else
  cache_status=1
  stop_server
fi
unset MEMRA_SERVE_SPEC

echo "== focused serve-smoke arm: Q35 mixed c=4 cold-prefill exactness =="
unset MEMRA_PRIME_BATCH MEMRA_PREFILL_TICK MEMRA_PRIME_BATCH_MAX_T \
  MEMRA_PRIME_BATCH_HOLD_MS
export MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_PREFIX_CACHE_MB=4096 \
  MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 \
  MEMRA_MAX_SESSIONS=96
if start_server "q35-coldfix=$Q35_MODEL" "$OUT_DIR/q35-server.log"; then
  (cd "$REPO_ROOT" && python3 tools/q35-cold-mixed-gate.py --base "$BASE" \
    --model q35-coldfix --namespace serve-smoke-q35-coldfix --timeout 600) \
    2>&1 | tee "$OUT_DIR/q35-gate.log"
  q35_gate_status=${PIPESTATUS[0]}
  stop_server
else
  q35_gate_status=1
  stop_server
fi
if grep -Eq '^\[prime-batch\].*carried=[1-9]' "$OUT_DIR/q35-server.log"; then
  q35_carried_status=1
  echo "FAIL: Q35 routed-MoE entered a carried prime batch"
else
  q35_carried_status=0
  echo "PASS: Q35 routed-MoE carried prime batches remain gated"
fi
unset MEMRA_SERVE_SPEC MEMRA_CTX MEMRA_PREFIX_CACHE_MB MEMRA_PREFIX_DEDUP \
  MEMRA_REUSE_POOL MEMRA_AFFINITY MEMRA_MAX_SESSIONS

nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true
date --iso-8601=seconds
echo "focused-arms: cache_meter_status=$cache_status"
echo "focused-arms: q35_gate_status=$q35_gate_status"
echo "focused-arms: q35_carried_status=$q35_carried_status"

[ "$cache_status" -eq 0 ] && [ "$q35_gate_status" -eq 0 ] && \
  [ "$q35_carried_status" -eq 0 ]
