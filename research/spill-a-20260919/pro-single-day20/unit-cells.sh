#!/usr/bin/env bash
# The door's GPU unit cells (option_b demote unwinds, option_c promote unwinds) plus the day-20 engine cell
# (`d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event`, memra-engine lib) on the target
# card, under the collector's lock hold: `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day20
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
git rev-parse HEAD > "$R/unit/tree.sha"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/cargo-test.log" 2>&1
rc=$?
echo "$rc" > "$R/unit/cargo-test.exit"
grep -E '^test |^test result' "$R/unit/cargo-test.log"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 d2d_capture_ > "$R/unit/cargo-test-engine.log" 2>&1
rc2=$?
echo "$rc2" > "$R/unit/cargo-test-engine.exit"
grep -E '^test |^test result' "$R/unit/cargo-test-engine.log"
echo "unit-cells rc=$rc engine-rc=$rc2"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ]
