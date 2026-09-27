#!/usr/bin/env bash
# The door's GPU unit cells (option_b demote unwinds with the day-30 and day-31 span cells, option_c promote unwinds with the day-32
# H2D span cells), the engine's D2D, D2H span and day-32 and day-33 H2D span cells (memra-engine lib), and the CPU cells (the hash, Hashing,
# span-order and day-31 censuses, the day-33 census, the memra-tier D2H and H2D span rules) on the target card
# and this host, under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day33
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?
echo "$rc" > "$R/unit/cargo-test.exit"
grep -E '^test |^test result' "$R/unit/cargo-test.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture d2d_ d2h_span h2d_span > "$R/unit/cargo-test-engine.log" 2>&1
rc2=$?
echo "$rc2" > "$R/unit/cargo-test-engine.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-engine.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting every_path_that_meets_a_promoting the_run_loop_waits_boundedly the_d2h_spans day31_ day33_ > "$R/unit/cargo-test-hash.log" 2>&1
rc3=$?
echo "$rc3" > "$R/unit/cargo-test-hash.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-hash.log"
cargo test -p memra-engine --lib -- span_rules_are_as_stated > "$R/unit/cargo-test-engine-census.log" 2>&1
rc4=$?
echo "$rc4" > "$R/unit/cargo-test-engine-census.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-engine-census.log"
cargo test -p memra-tier --test contracts -- d2h_span h2d_span > "$R/unit/cargo-test-tier.log" 2>&1
rc5=$?
echo "$rc5" > "$R/unit/cargo-test-tier.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-tier.log"
echo "unit-cells rc=$rc engine-rc=$rc2 hash-rc=$rc3 engine-census-rc=$rc4 tier-rc=$rc5"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
