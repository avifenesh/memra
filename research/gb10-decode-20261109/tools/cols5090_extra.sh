#!/usr/bin/env bash
# Rig: lever 3a's extra exactness cells on the RTX 5090 (the review's uncovered fill sites), one
# gpulock lease per cell:
#   seam-unpacked / seam-rowwise: run-spec K=1..8 on Qwen3.8-27B with MEMRA_SPEC_GDN_PACKED=0 and
#     with MEMRA_SPEC_VERIFY_ROWWISE=1 (the per-row and rowwise ColStates fills); since the
#     seam's removal (2026-11-09) seam-unpacked refuses, and its receipt is the 2026-11-09 run;
#   pool-B / pool-T: research/batch-spec-20261101/battery/pool_cell.py (q9, c = 2, 4, 8 pooled
#     cohorts, greedy and sampled) on the stack parent and on the head; every stream must match.
# usage: cols5090_extra.sh HEAD_SHA [BASE_SHA]   (the base is the stack parent, lever 1 on main)
set -u
export PATH=/usr/bin:$PATH
D=$HOME/.cache/memra-gb10-r4
H=$1
B=$D/cells-work/rtx5090/bins/${2:-e0887334a04c385a450fbdfcd3114f523624935d}-120a
T=$D/cells-work/rtx5090/bins/$H-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
CD=$D/scout/cols-extra-5090; mkdir -p "$CD"
L=$CD/driver.log
W=$D/wt-cols/research/batch-spec-20261101/battery
lease() { # NAME CMD...
  local name=$1; shift
  echo "$(date -Is) $name start" >> "$L"
  gpulock run --ttl 20m --for "memra round 5 lever 3a exactness cell on the 5090: $name (one cell per lease)" --wait 60m -- "$@" >> "$L" 2>&1
  echo "$(date -Is) $name rc=$?" >> "$L"
}
echo "$(date -Is) seam-unpacked refused: MEMRA_SPEC_GDN_PACKED was removed 2026-11-09 (docs/FLAGS.md Removed doors), its per-row arm is gone" | tee -a "$L" >&2
lease seam-rowwise bash -c "MEMRA_SPEC_VERIFY_ROWWISE=1 '$T/run-spec' '$Q38' > '$CD/run-spec-rowwise.log' 2>&1; tail -n 1 '$CD/run-spec-rowwise.log'"
lease pool-B /usr/bin/python3 "$W/pool_cell.py" "$B/memra-server" "$CD/pool-B" 8351
lease pool-T /usr/bin/python3 "$W/pool_cell.py" "$T/memra-server" "$CD/pool-T" 8351
/usr/bin/python3 "$W/compare_pool_runs.py" "$CD/pool-B" head="$CD/pool-T" > "$CD/compare-pool.txt" 2>&1
echo "$(date -Is) compare rc=$?" >> "$L"
echo "$(date -Is) all cells done" >> "$L"
