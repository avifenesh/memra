#!/usr/bin/env bash
# lane-8 interleaved A/B driver (house perf law: alternating runs x5, same box +
# clock window, medians banked from the per-run JSONs; no cross-window comparisons).
#
# Two tables:
#   seam  : tip binary, MEMRA_DSV4_DECODE_PATH legacy vs device  (path attribution)
#   lane7 : lane-7-tip binary (no seam) vs tip binary device      (the headline)
#
# Usage (on the box):
#   ./ab_interleaved.sh seam  <n_new> <reps>
#   ./ab_interleaved.sh lane7 <n_new> <reps>   # needs ~/memra-lane7 build (see below)
#
# lane-7 baseline build (once). 77ef38924a = lane-7 kernels + the lane-8 bench
# harness ONLY (the harness commit precedes every kernel change) — the honest
# "lane-7 baseline" binary, since 1dc08d00f3 itself has no bench binary:
#   cd ~/memra-src && git worktree add ~/memra-lane7 77ef38924a
#   cd ~/memra-lane7 && MEMRA_CUDA_ARCH=120a ~/.cargo/bin/cargo build --release \
#       -j 20 -p memra-engine --bin dsv4-decode-bench
set -euo pipefail
MODE="${1:?seam|lane7}"
N_NEW="${2:-1024}"
REPS="${3:-5}"
MODEL=/home/ubuntu/models/dsv4-flash-nvfp4
FIX=/home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json
OUT=/home/ubuntu/dsv4-lane8-out/ab-$MODE-$N_NEW
mkdir -p "$OUT"
TIP_BIN=/home/ubuntu/memra-src/target/release/dsv4-decode-bench
L7_BIN=/home/ubuntu/memra-lane7/target/release/dsv4-decode-bench

clocks() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader | tr '\n' '|'; }

run_one() { # label bin path_seam idx
  local label="$1" bin="$2" seam="$3" i="$4"
  echo "[$(date -u +%T)] run $label#$i clocks: $(clocks)" | tee -a "$OUT/ab.log"
  MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DECODE_PATH="$seam" \
    "$bin" "$MODEL" "$FIX" "$OUT/$label-$i.json" "$N_NEW" 0,1 \
    2>&1 | grep -E 'window|sha256|decode path' | tee -a "$OUT/ab.log"
}

for i in $(seq 1 "$REPS"); do
  case "$MODE" in
    seam)
      run_one A-legacy "$TIP_BIN" legacy "$i"
      run_one B-device "$TIP_BIN" device "$i"
      ;;
    lane7)
      run_one A-lane7 "$L7_BIN" "" "$i"
      run_one B-device "$TIP_BIN" device "$i"
      ;;
  esac
done
echo "A/B $MODE complete -> $OUT" | tee -a "$OUT/ab.log"
