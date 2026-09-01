#!/usr/bin/env bash
# Default-peer no-regression receipt on the designated 2x PRO 6000 verification box.
set -euo pipefail

ROOT=${ROOT:-$HOME/memra-cx-hostbounce}
OUT=${OUT:-$HOME/hostbounce-box1}
MODEL=${MODEL:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1
exec 9>/tmp/memra-gpu.lock
flock -w 1800 9

cd "$ROOT"
source "$HOME/.cargo/env"
export CUDA_HOME=/usr/local/cuda-13.2

echo "lock_acquired=$(date -u +%FT%TZ)"
echo "host=$(hostname)"
echo "source_commit=$(git rev-parse HEAD)"
echo "source_status_begin"
git status --short --branch
echo "source_status_end"
sha256sum target/release/decode-batch-gate
stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,pstate,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-before.csv"
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
if [[ -n "$apps" ]]; then
    echo "$apps"
    echo "FAIL: GPU was not idle at lock acquisition"
    exit 1
fi

set +e
env -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PP_SHARD \
    MEMRA_PP_DEVICES=0,1 \
    timeout 7200 target/release/decode-batch-gate "$MODEL" \
        --mode pp --batch 1,2,4,8 --steps 24 --reps 2 --stages 2 --plen 520 \
        2>&1 | tee "$OUT/decode-batch-gate.log"
gate_rc=${PIPESTATUS[0]}
set -e

echo "decode_batch_gate_rc=$gate_rc"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,pstate,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-after.csv"
echo "lock_released=$(date -u +%FT%TZ)"
exit "$gate_rc"
