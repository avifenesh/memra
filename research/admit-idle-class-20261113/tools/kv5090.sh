#!/usr/bin/env bash
# memra#1304 review fold, runs on the RTX 5090 rig under `gpulock run` (balanced profile; the cell
# is pass/fail, no timing): tools/native_choices.py --phase kv with the accounting observer built
# from the fold commit, then its verifier. The lease gives FD 9 on the memra 5090 lock and
# MEMRA_GPU_LOCK; the tool refuses without them.
# usage: kv5090.sh WORKTREE OBSERVER_BINARY SOURCE_SHA OUT_DIR
set -euo pipefail
WT=$1; BIN=$2; SRC=$3; OUT=$4
MODEL=/data/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf
MODEL_SHA=0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c
L=$(dirname "$OUT")
mkdir -p "$L"
# The tool records the out path it is given (the metadata file's path in env.json): run from the
# receipt's parent with a relative path, so the receipt carries no home directory.
cd "$L"; OUT=$(basename "$OUT")
[ -x "$BIN" ] || { echo "observer binary missing: not run" | tee "$L/kv.log"; exit 4; }
[ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ] || { echo "not on AC: not run" | tee "$L/kv.log"; exit 3; }
export CUDA_VISIBLE_DEVICES=0
{ date -Is; cat /sys/class/power_supply/ADP0/online; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$L/gpu-before.txt" 2>&1
python3 "$WT/tools/native_choices.py" --binary "$BIN" --binary-sha "$(sha256sum "$BIN" | cut -d' ' -f1)" \
  --model "$MODEL" --model-sha "$MODEL_SHA" --source "$SRC" --out "$OUT" --phase kv --port 18275 2>&1 | tee "$L/kv.log"
{ date -Is; cat /sys/class/power_supply/ADP0/online; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$L/gpu-after.txt" 2>&1
( cd "$WT/tools" && python3 replay_native_choices.py "$L/$OUT" ) 2>&1 | tee "$L/verify.log"
