#!/usr/bin/env bash
# The door's GPU unit cells (option_b demote unwinds, option_c promote unwinds), the day-20 to day-22 engine cells
# (`d2d_capture_*`, `d2d_restore_*`, `d2d_receipt_*`, `d2d_early_reader_*`, memra-engine lib) on the target card, and
# the day-28 and day-29 CPU hash cells (`hash_*`, `hashing_*`, the Hashing census, the parked-only wait) on this host, under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day29
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?
echo "$rc" > "$R/unit/cargo-test.exit"
grep -E '^test |^test result' "$R/unit/cargo-test.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture d2d_ > "$R/unit/cargo-test-engine.log" 2>&1
rc2=$?
echo "$rc2" > "$R/unit/cargo-test-engine.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-engine.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting the_run_loop_waits_boundedly > "$R/unit/cargo-test-hash.log" 2>&1
rc3=$?
echo "$rc3" > "$R/unit/cargo-test-hash.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-hash.log"
echo "unit-cells rc=$rc engine-rc=$rc2 hash-rc=$rc3"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
