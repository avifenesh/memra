#!/usr/bin/env bash
# OWED 26 correction cells (M1-PREREG G, after the failed serving-shape check), under the canonical lock.
# Red: the e5d899500 build with only the two correction cells added must FAIL both (each waits 30 s).
# Green: the corrected build must PASS the four OWED 26 cells and every pool GPU cell.
# Usage: run-cells2.sh OUT_DIR RED_TEST_BIN GREEN_TEST_BIN
set -u
out=$1; red=$2; green=$3
mkdir -p "$out"
sha256sum "$red" "$green" > "$out/binaries.sha256"
a=spill_pread::tests::demand_wait_with_a_free_buffer_and_nothing_in_flight_returns_at_once
b=spill_pread::tests::demand_wait_after_every_h2d_event_completed_returns_at_once
"$red" --ignored --exact "$a" --nocapture > "$out/red-$a.log" 2>&1; ra=$?
"$red" --ignored --exact "$b" --nocapture > "$out/red-$b.log" 2>&1; rb=$?
"$green" --ignored spill_pread::tests --nocapture --test-threads 1 > "$out/green-spill_pread-gpu-cells.log" 2>&1; g=$?
printf '{"red_free_rc": %d, "red_reaped_rc": %d, "green_pool_gpu_cells_rc": %d}\n' "$ra" "$rb" "$g" > "$out/result.json"
cat "$out/result.json"
[ "$ra" -ne 0 ] && [ "$rb" -ne 0 ] && [ "$g" -eq 0 ]
