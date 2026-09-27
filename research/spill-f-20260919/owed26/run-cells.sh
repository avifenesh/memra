#!/usr/bin/env bash
# OWED 26 G2 GPU cells (M1-PREREG G2). Runs under the canonical lock (the caller holds it).
# Red arm: the pre-fix build with only the first cell added must FAIL that cell.
# Green: the fix build must PASS both new cells and the pool's existing GPU cells.
# Usage: run-cells.sh OUT_DIR RED_TEST_BIN GREEN_TEST_BIN
set -u
out=$1; red=$2; green=$3
mkdir -p "$out"
sha256sum "$red" "$green" > "$out/binaries.sha256"
t1=spill_pread::tests::demand_submit_waits_for_a_buffer_instead_of_returning_ring_busy
t2=spill_pread::tests::demand_submit_returns_none_only_when_prefetches_hold_every_buffer
"$red" --ignored --exact "$t1" --nocapture > "$out/red-$t1.log" 2>&1; red_rc=$?
"$green" --ignored --exact "$t1" --nocapture > "$out/green-$t1.log" 2>&1; g1=$?
"$green" --ignored --exact "$t2" --nocapture > "$out/green-$t2.log" 2>&1; g2=$?
"$green" --ignored spill_pread::tests --nocapture --test-threads 1 > "$out/green-spill_pread-gpu-cells.log" 2>&1; g3=$?
printf '{"red_rc": %d, "green_waits_rc": %d, "green_held_rc": %d, "green_pool_gpu_cells_rc": %d}\n' \
  "$red_rc" "$g1" "$g2" "$g3" > "$out/result.json"
cat "$out/result.json"
# Pass only when the red arm failed and every green cell passed.
[ "$red_rc" -ne 0 ] && [ "$g1" -eq 0 ] && [ "$g2" -eq 0 ] && [ "$g3" -eq 0 ]
