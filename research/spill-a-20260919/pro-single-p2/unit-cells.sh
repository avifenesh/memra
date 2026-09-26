#!/usr/bin/env bash
# The design-P2 target-card unit cells on the tip tree (DAY52 section 1 (a); the design-V set of DAY47), under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`. The test binaries were
# built by build.sh outside the hold. The engine's native tier_transfer cells in ONE process at the default thread count,
# three runs (each owns a pool context), then once serially; the door's GPU cells (serial); the CPU censuses (the hash,
# Hashing and Demoting censuses, days 31 to 41); the engine's rule censuses with the one-kernel-stream census and the
# context-pool census; the tier's span, H2D, device-receipt and D2D rules.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-p2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
green=0
for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --nocapture tier_transfer::tests:: > "$R/unit/engine-parallel-$r.log" 2>&1
  rc=$?; [ $rc -eq 0 ] && green=$((green + 1))
  echo "parallel run $r rc=$rc $(grep -h '^test result' "$R/unit/engine-parallel-$r.log")" | tee -a "$R/unit/run.log"
done
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture tier_transfer::tests:: > "$R/unit/engine-serial.log" 2>&1
rc2=$?; grep -E '^test |^test result|D2H DEVICE RECEIPT' "$R/unit/engine-serial.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/door-cells.log" 2>&1
rc=$?; grep -E '^test |^test result' "$R/unit/door-cells.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting every_path_that_meets_a_promoting the_run_loop_waits_boundedly the_d2h_spans day31_ day33_ day34_ day35_ day36_ day38_ day41_ day42_ day47_ day49_ day51_ day52_ pause_ > "$R/unit/cpu-censuses.log" 2>&1
rc3=$?; grep -E '^test result' "$R/unit/cpu-censuses.log"
cargo test -p memra-engine --lib -- rules_are_as_stated native_cells_own_their_context one_side_stream_beside_the_owner span_receipt_rules day39_ > "$R/unit/engine-censuses.log" 2>&1
rc4=$?; grep -E '^test |^test result' "$R/unit/engine-censuses.log"
cargo test -p memra-tier --test contracts -- d2h_span h2d_ d2h_device_receipt d2d_ span_receipt > "$R/unit/tier.log" 2>&1
rc5=$?; grep -E '^test result' "$R/unit/tier.log"
echo "unit-cells parallel=$green/3 engine-serial-rc=$rc2 door-rc=$rc cpu-rc=$rc3 engine-census-rc=$rc4 tier-rc=$rc5" | tee -a "$R/unit/run.log"
[ $green -eq 3 ] && [ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
