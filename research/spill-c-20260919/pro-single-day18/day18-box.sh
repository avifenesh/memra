#!/usr/bin/env bash
# Day 18 BOX3 cells in order: hashlock, hashmicro, overlap, serverdoor (one lock hold each, bounded retries).
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export D18_RIG=pro-single D18_R=/root/spill-receipts/c-day18 D18_TREE=/root/wt-c
export D18_CELL_SCRIPT=/root/spill-receipts/c-day18/day18-cell.sh D18_BINS=/root/spill-receipts/c-day18/bins
export D18_LOCK=/tmp/memra-gpu.lock D18_PORT=18141 D18_MOE_ENV="MEMRA_MOE_RESIDENT=0"
export D18_ART=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
export D18_ART_OTHER=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
echo "box start $(date -u +%FT%TZ) tree $(git -C /root/wt-c rev-parse HEAD)"
sha256sum "$D18_CELL_SCRIPT" /root/spill-receipts/c-day18/day18-run-cell.sh
for spec in hashlock:1500 hashmicro:900 overlap:3600 serverdoor:1500; do
  cell=${spec%%:*}; to=${spec##*:}
  bash /root/spill-receipts/c-day18/day18-run-cell.sh "$cell" "$to"; echo "$cell rc=$? $(date -u +%FT%TZ)"
  python3 /root/wt-c/tools/tier-battery.py --rig pro-single --validate "$D18_R/$cell" > "$D18_R/$cell-validate.log" 2>&1; echo "$cell validate rc=$?"
  touch "$D18_R/$cell.done"
done
echo "box done $(date -u +%FT%TZ)"
