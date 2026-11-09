#!/usr/bin/env bash
# Rig: the review's S1 cells for the GDN packed seam removal on the RTX 5090, one gpulock lease
# per cell, binaries prebuilt outside the leases (the review-fix head). They cover verify-graph
# replays at odd T, where the 3-entry table's alt entry is written and canonical/alt swap.
#   run-spec-ornith-k{2,4,6,8}: MEMRA_SPEC_K=k run-spec on Ornith-1.5-35B-A3B in a fresh process,
#     so the MTP verify-graph pool is built at vt_cap = k+1 (odd) and full rounds replay at odd T.
#     MEMRA_DSPARK_FULLG_DEBUG=1 prints the first graph-walk round's t and t_cap (diagnostic).
#   dspark-q38-vg: MEMRA_DSPARK_VERIFY_GRAPH=1 dspark_q38_gate on Qwen3.8-27B with the DFlash2
#     drafter: the reactive window ladder puts mixed widths through one pool; ALL EXACT required.
# usage: gpu-cells-2.sh   (after rig-build-2.sh)
set -u
export PATH=/usr/bin:$PATH
S=$HOME/.cache/memra-gb10-r5/seam
W=$HOME/.cache/memra-gb10-r5/wt-seam
T=$S/bins/0a7d83bf65-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
DRAFT=/data/ai-ml/models/q38-dflash2
cd "$W" || exit 2
CD=$S/cells-2; mkdir -p "$CD"
L=$CD/driver.log
yield_landing() {
  waitfor --quiet --until "! gpulock status | grep -qi landing" --every 30s --timeout 60m >> "$L" 2>&1
  # start a cell only on a card under 75 C (keeps it usable for others; the cells are pass/fail)
  waitfor --quiet --until "bash $W/research/gb10-decode-20261109/tools/gdn_seam_cool.sh" --every 15s --timeout 30m >> "$L" 2>&1
  echo "$(date -Is) card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$L"
}
lease() { # NAME TTL LOG CMD...
  local name=$1 ttl=$2 log=$3; shift 3
  yield_landing
  echo "$(date -Is) $name start; $(gpulock status 2>&1 | head -n 3 | tr '\n' ' ')" >> "$L"
  gpulock run --ttl "$ttl" --for "memra GDN packed seam removal: $name (one cell per lease)" --wait 90m -- "$@" > "$log" 2>&1
  echo "$(date -Is) $name rc=$? last: $(tail -n 1 "$log")" >> "$L"
}
for k in 2 4 6 8; do
  lease "run-spec K=$k on Ornith-1.5-35B-A3B, verify-graph pool at vt_cap=$((k + 1))" 15m "$CD/run-spec-ornith-k$k.log" \
    env MEMRA_SPEC_K=$k MEMRA_DSPARK_FULLG_DEBUG=1 "$T/run-spec" "$ORN"
done
lease "dspark_q38_gate with MEMRA_DSPARK_VERIFY_GRAPH=1 on Qwen3.8-27B" 20m "$CD/dspark-q38-vg.log" \
  env MEMRA_DSPARK_VERIFY_GRAPH=1 MEMRA_DSPARK_FULLG_DEBUG=1 "$T/dspark_q38_gate" "$Q38" "$DRAFT"
echo "$(date -Is) all cells done" >> "$L"
