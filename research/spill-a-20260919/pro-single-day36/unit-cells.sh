#!/usr/bin/env bash
# Day-36 target-card unit cells on the tip tree (DAY36 section 3), under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`. The test binaries
# were built by build.sh outside the hold. The door's GPU cells (option_b_, option_c_, with day 35's changed-lease cell),
# the engine's D2D, D2H span and H2D cells (serial), the CPU censuses (days 31, 33, 34, 35 and 36 with the hash and
# Hashing censuses), the engine's rule censuses and the tier's D2H span and H2D rules.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day36
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?; echo "$rc" > "$R/unit/cargo-test.exit"; grep -E '^test |^test result' "$R/unit/cargo-test.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture d2d_ d2h_span h2d_ > "$R/unit/cargo-test-engine.log" 2>&1
rc2=$?; echo "$rc2" > "$R/unit/cargo-test-engine.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-engine.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting every_path_that_meets_a_promoting the_run_loop_waits_boundedly the_d2h_spans day31_ day33_ day34_ day35_ day36_ > "$R/unit/cargo-test-hash.log" 2>&1
rc3=$?; echo "$rc3" > "$R/unit/cargo-test-hash.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-hash.log"
cargo test -p memra-engine --lib -- rules_are_as_stated > "$R/unit/cargo-test-engine-census.log" 2>&1
rc4=$?; echo "$rc4" > "$R/unit/cargo-test-engine-census.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-engine-census.log"
cargo test -p memra-tier --test contracts -- d2h_span h2d_ > "$R/unit/cargo-test-tier.log" 2>&1
rc5=$?; echo "$rc5" > "$R/unit/cargo-test-tier.exit"; grep -E '^test |^test result' "$R/unit/cargo-test-tier.log"
echo "unit-cells rc=$rc engine-rc=$rc2 hash-rc=$rc3 engine-census-rc=$rc4 tier-rc=$rc5"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
