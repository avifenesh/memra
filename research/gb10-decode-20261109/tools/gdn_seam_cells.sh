#!/usr/bin/env bash
# Rig: the GDN packed seam removal's GPU cells on the RTX 5090, one gpulock lease per cell,
# binaries prebuilt outside the leases. Before each lease it yields to a waiter whose reason
# says "landing".
#   run-spec-q38 / run-spec-ornith: run-spec K=1..8 self-consistency on the two roster models
#   pool-B / pool-T: research/batch-spec-20261101/battery/pool_cell.py on the stack parent's
#     server and on this branch's; every stream must match
#   kernel-check: the battery's form (first roster model, both required manifests)
# usage: gpu-cells.sh   (after rig-build.sh has built the bins)
set -u
export PATH=/usr/bin:$PATH
S=$HOME/.cache/memra-gb10-r5/seam
W=$HOME/.cache/memra-gb10-r5/wt-seam
T=$S/bins/ed2e0c85ac-120a
B=$HOME/.cache/memra-gb10-r4/cells-work/rtx5090/bins/3ab0a686a1668d4bcdc842774455333891c61a2f-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
cd "$W" || exit 2
CD=$S/cells; mkdir -p "$CD"
L=$CD/driver.log
PB=$W/research/batch-spec-20261101/battery
yield_landing() {
  waitfor --quiet --until "! gpulock status | grep -qi landing" --every 30s --timeout 60m >> "$L" 2>&1
}
lease() { # NAME TTL LOG CMD...
  local name=$1 ttl=$2 log=$3; shift 3
  yield_landing
  echo "$(date -Is) $name start; $(gpulock status 2>&1 | head -n 3 | tr '\n' ' ')" >> "$L"
  gpulock run --ttl "$ttl" --for "memra GDN packed seam removal: $name (one cell per lease)" --wait 90m -- "$@" > "$log" 2>&1
  echo "$(date -Is) $name rc=$? last: $(tail -n 1 "$log")" >> "$L"
}
lease "run-spec K=1..8 on Qwen3.8-27B" 15m "$CD/run-spec-q38.log" "$T/run-spec" "$Q38"
lease "run-spec K=1..8 on Ornith-1.5-35B-A3B" 15m "$CD/run-spec-ornith.log" "$T/run-spec" "$ORN"
lease "pooled-cohort stream cell, stack parent server" 20m "$CD/pool-B.log" /usr/bin/python3 "$PB/pool_cell.py" "$B/memra-server" "$CD/pool-B" 8351
lease "pooled-cohort stream cell, this branch's server" 20m "$CD/pool-T.log" /usr/bin/python3 "$PB/pool_cell.py" "$T/memra-server" "$CD/pool-T" 8351
/usr/bin/python3 "$PB/compare_pool_runs.py" "$CD/pool-B" head="$CD/pool-T" > "$CD/compare-pool.txt" 2>&1
echo "$(date -Is) compare rc=$? $(tail -n 1 "$CD/compare-pool.txt")" >> "$L"
lease "kernel-check (copy_indirect_src_f32 removed)" 30m "$CD/kernel-check.log" \
  env -u MEMRA_KC_FAST -u MEMRA_KC_ONLY "$T/kernel-check" "$ORN" \
  --require-manifest "$W/tools/kernel-check-27b.cells" --require-manifest "$W/tools/kernel-check-step35.cells"
echo "$(date -Is) all cells done" >> "$L"
