#!/usr/bin/env bash
set -euo pipefail
EV="$PWD/research/modelplan-onboarding-20261002-qwen35-9b/receipts"
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
TELEMETRY_PID=
cleanup() {
  if [ -n "$TELEMETRY_PID" ]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -lms 250 > "$EV/gpu-telemetry.csv" &
TELEMETRY_PID=$!
sha256sum -c "$EV/build/binaries.sha256"
sha256sum -c "$EV/build/model.sha256"
target/release/memra model inspect "$MODEL" --against qwen35 --out "$EV/inspect" > "$EV/inspect.log" 2>&1
set +e
target/release/memra model verify serve "$MODEL" --against qwen35 --out "$EV/inspect" --native-runner "$PWD/target/release/memra-server" > "$EV/strict-serve-refusal.log" 2>&1
STRICT_RC=$?
set -e
[ "$STRICT_RC" -eq 2 ]
/bin/grep -q 'checkpoint-parity.tsv' "$EV/strict-serve-refusal.log"
python3 tools/collect-serving-qualification.py --model "$MODEL" --server "$PWD/target/release/memra-server" --cli "$PWD/target/release/memra" --inspection "$EV/inspect" --port 18120 --external-lock 9 --out "$EV/serving-v1"
