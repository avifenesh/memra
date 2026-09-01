#!/usr/bin/env bash
# Final-head controller state/continuation identity on the real serving prompt.
set -euo pipefail

ROOT=${OPTI2_ROOT:-/home/ubuntu/memra-opti2}
OUT=${OPTI2_FINAL_HIT_OUT:-/home/ubuntu/opti2-receipts/final-hit-state-1}
NGEN=${OPTI2_FINAL_HIT_NGEN:-128}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
GATE=${ROOT}/target/release/optipipe-gate

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "FINAL_HIT_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch

/home/ubuntu/.cargo/bin/cargo build --release -p memra-engine --bin optipipe-gate \
    2>&1 | tee "$OUT/build.log"
sha256sum "$GATE" > "$OUT/SHA256SUMS"

nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,temperature.gpu,pstate,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/nvidia-smi-before.log" 2>&1
apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

filler='The quick brown fox jumps over the lazy dog while the seasoned engineer measures throughput, latency, and saturation across every replica. '
prompt='Summarize the operational state of a GPU serving cluster in exactly three sentences, then list four risks. Context follows. '
for _ in {1..8}; do
    prompt+="$filler"
done

env \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_SPEC_GATE=0 \
    MEMRA_SPEC_K=1 \
    MEMRA_SPEC_STATS=1 \
    MEMRA_SPEC_DEVACC=1 \
    MEMRA_MTP_DRAFT="$DRAFT" \
    MEMRA_OPTI_CONTROLLER_Q=0.0 \
    MEMRA_OPTI_PROMPT="$prompt" \
    MEMRA_CHAT=1 \
    MEMRA_NGEN="$NGEN" \
    "$GATE" "$MODEL" controller \
    2>&1 | tee "$OUT/controller-real-prompt.log"

python3 - "$OUT/controller-real-prompt.log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
if "STATE IDENTITY: PASS mode=controller" not in text:
    raise SystemExit("missing controller state-identity pass marker")
match = re.search(r"stats=OptiForkGateStats \{ attempts: (\d+), hits: (\d+), misses: (\d+)", text)
if not match:
    raise SystemExit("missing controller counters")
attempts, hits, misses = map(int, match.groups())
if attempts == 0 or hits == 0 or misses == 0 or hits + misses != attempts:
    raise SystemExit(
        f"controller did not exercise both terminal paths: attempts={attempts} hits={hits} misses={misses}"
    )
print(f"FINAL_HIT_COUNTERS attempts={attempts} hits={hits} misses={misses}")
PY

nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,temperature.gpu,pstate,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/nvidia-smi-after.log" 2>&1
apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "FINAL_HIT_STATE_PASS $(date -u +%FT%TZ)"
