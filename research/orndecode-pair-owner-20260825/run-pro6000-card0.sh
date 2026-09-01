#!/usr/bin/env bash
set -euo pipefail

MODEL=/opt/dl-image/nvme/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
BASE_BIN=/opt/dl-image/nvme/wt-ornith-pair-owner-base-20260825/target/release/decode-batch-bench
CAND_BIN=/opt/dl-image/nvme/wt-ornith-pair-owner-20260825/target/release/decode-batch-bench
OUT=/opt/dl-image/nvme/wt-ornith-pair-owner-20260825/research/orndecode-pair-owner-20260825/raw/pro6000-card0/perf
EXPECTED_MODEL_SHA=72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3

mkdir -p "$OUT"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo "gpu lock held" >&2; exit 75; }

nvidia-smi -i 0 --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader > "$OUT/gpu-before.csv"
nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$OUT/compute-apps-before.csv"
if [ -s "$OUT/compute-apps-before.csv" ]; then
  echo "compute app present before scored window" >&2
  sed -n '1,20p' "$OUT/compute-apps-before.csv" >&2
  exit 1
fi

test -x "$BASE_BIN"
test -x "$CAND_BIN"
test -f "$MODEL"
actual_model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
if [ "$actual_model_sha" != "$EXPECTED_MODEL_SHA" ]; then
  echo "model sha mismatch: $actual_model_sha" >&2
  exit 1
fi
sha256sum "$BASE_BIN" "$CAND_BIN" > "$OUT/binary-sha256.txt"
printf 'model_sha256=%s\n' "$actual_model_sha" > "$OUT/run-meta.txt"
printf 'base_source=%s\n' "$(git -C /opt/dl-image/nvme/wt-ornith-pair-owner-base-20260825 rev-parse HEAD)" >> "$OUT/run-meta.txt"
printf 'candidate_source=%s\n' "$(git -C /opt/dl-image/nvme/wt-ornith-pair-owner-20260825 rev-parse HEAD)" >> "$OUT/run-meta.txt"
printf 'candidate_diff_sha256=%s\n' "$(git -C /opt/dl-image/nvme/wt-ornith-pair-owner-20260825 diff --binary | sha256sum | awk '{print $1}')" >> "$OUT/run-meta.txt"

telemetry_pid=
cleanup_telemetry() {
  if [ -n "$telemetry_pid" ]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    telemetry_pid=
  fi
}
trap cleanup_telemetry EXIT INT TERM

run_arm() {
  local tag=$1
  local binary=$2
  local batches=$3
  local raw="$OUT/$tag.log"
  local telemetry="$OUT/$tag.telemetry.csv"

  nvidia-smi -i 0 \
    --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv -lms 250 > "$telemetry" 2>&1 &
  telemetry_pid=$!
  set +e
  env CUDA_VISIBLE_DEVICES=0 MEMRA_EXACT16_MOE=1 MEMRA_MOE_PAIR_OWNER=1 \
    "$binary" "$MODEL" --steps 128 --reps 5 --batches "$batches" --ctx 512 \
    > "$raw" 2>&1
  local rc=$?
  set -e
  cleanup_telemetry
  if [ "$rc" -ne 0 ]; then
    echo "$tag failed rc=$rc" >&2
    tail -40 "$raw" >&2
    exit "$rc"
  fi
  grep -E '^(loaded|B=|scale B=)' "$raw" | sed "s/^/$tag /"
}

# ABBA across binaries; reverse the per-process batch order in the BA half.
run_arm base-a "$BASE_BIN" 8,16
run_arm cand-a "$CAND_BIN" 8,16
run_arm cand-b "$CAND_BIN" 16,8
run_arm base-b "$BASE_BIN" 16,8

nvidia-smi -i 0 --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader > "$OUT/gpu-after.csv"
nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$OUT/compute-apps-after.csv"
if [ -s "$OUT/compute-apps-after.csv" ]; then
  echo "compute app remained after scored window" >&2
  sed -n '1,20p' "$OUT/compute-apps-after.csv" >&2
  exit 1
fi

echo "PERF-DONE"
