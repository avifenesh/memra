#!/usr/bin/env bash
# Day-38 target-card unit cells on the tip tree (DAY38 section 4), under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`. The test binaries were
# built by build.sh outside the hold. DAY37 section 1's target-card clause first: the engine's thirteen native tier_transfer
# cells in ONE process at the default thread count, 20 runs (the all arm), then once serially; then the door's GPU cells
# (serial), the CPU censuses (days 31 to 38 with the hash, Hashing and Demoting censuses), the engine's rule censuses and
# the context-pool census, and the tier's span, H2D and device-receipt rules.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day38
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit/all-arm"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
green=0
for r in $(seq 1 20); do
  CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --nocapture tier_transfer::tests:: > "$R/unit/all-arm/all-$r.log" 2>&1
  rc=$?; [ $rc -eq 0 ] && green=$((green + 1))
  echo "all run $r rc=$rc $(grep -h '^test result' "$R/unit/all-arm/all-$r.log")" | tee -a "$R/unit/all-arm/run.log"
done
echo "DAY37 FINDING5 TARGET all-arm green=$green of 20 rule 20 of 20 -> $([ $green -eq 20 ] && echo PASS || echo FAIL)" | tee -a "$R/unit/all-arm/run.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture tier_transfer::tests:: > "$R/unit/cargo-test-engine-serial.log" 2>&1
rc2=$?; echo "$rc2" > "$R/unit/cargo-test-engine-serial.exit"; grep -E '^test |^test result|D2H DEVICE RECEIPT' "$R/unit/cargo-test-engine-serial.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?; echo "$rc" > "$R/unit/cargo-test.exit"; grep -E '^test |^test result' "$R/unit/cargo-test.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting every_path_that_meets_a_promoting the_run_loop_waits_boundedly the_d2h_spans day31_ day33_ day34_ day35_ day36_ day38_ > "$R/unit/cargo-test-hash.log" 2>&1
rc3=$?; echo "$rc3" > "$R/unit/cargo-test-hash.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-hash.log"
cargo test -p memra-engine --lib -- rules_are_as_stated native_cells_own_their_context > "$R/unit/cargo-test-engine-census.log" 2>&1
rc4=$?; echo "$rc4" > "$R/unit/cargo-test-engine-census.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-engine-census.log"
cargo test -p memra-tier --test contracts -- d2h_span h2d_ d2h_device_receipt > "$R/unit/cargo-test-tier.log" 2>&1
rc5=$?; echo "$rc5" > "$R/unit/cargo-test-tier.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-tier.log"
echo "unit-cells all-arm=$green/20 engine-serial-rc=$rc2 door-rc=$rc hash-rc=$rc3 engine-census-rc=$rc4 tier-rc=$rc5"
[ $green -eq 20 ] && [ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
