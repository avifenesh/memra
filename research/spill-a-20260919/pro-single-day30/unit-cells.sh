#!/usr/bin/env bash
# The door's GPU unit cells (option_b demote unwinds with the day-30 span cells, option_c promote unwinds), the day-20 to
# day-22 engine cells (`d2d_*`) and the day-30 span cell (`d2h_span_*`, memra-engine lib) on the target card, and the
# day-28 to day-30 CPU cells (`hash_*`, `hashing_*`, the Hashing census, the parked-only wait, the span-order census, the
# memra-tier span rules) on this host, under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day30
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?
echo "$rc" > "$R/unit/cargo-test.exit"
grep -E '^test |^test result' "$R/unit/cargo-test.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture d2d_ d2h_span > "$R/unit/cargo-test-engine.log" 2>&1
rc2=$?
echo "$rc2" > "$R/unit/cargo-test-engine.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-engine.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting the_run_loop_waits_boundedly the_d2h_spans > "$R/unit/cargo-test-hash.log" 2>&1
rc3=$?
echo "$rc3" > "$R/unit/cargo-test-hash.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-hash.log"
cargo test -p memra-tier --test contracts -- d2h_span > "$R/unit/cargo-test-tier.log" 2>&1
rc4=$?
echo "$rc4" > "$R/unit/cargo-test-tier.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-tier.log"
echo "unit-cells rc=$rc engine-rc=$rc2 hash-rc=$rc3 tier-rc=$rc4"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ]
