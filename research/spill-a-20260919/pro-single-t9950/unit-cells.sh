#!/usr/bin/env bash
# DAY44 section 1: design T's unit cells on this 9950X-class host (DAY39 section 5's (c)), ft's prebuilt test binaries
# (built by build.sh on the arms' crates), under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`. The engine's
# native tier_transfer cells in one process, then serially; the door's GPU cells (serial); the CPU censuses of the ft tree.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-t9950
cd /root/wt-a || exit 1
mkdir -p "$R/unit"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/unit/LOCK.json"
sha256sum "$R"/bins/ft/*-tests > "$R/unit/binaries.sha256"
(cd crates/memra-engine && CUDA_VISIBLE_DEVICES=0 "$R/bins/ft/memra-engine-tests" --ignored --nocapture tier_transfer::tests:: > "$R/unit/engine-parallel.log" 2>&1); rc1=$?
(cd crates/memra-engine && CUDA_VISIBLE_DEVICES=0 "$R/bins/ft/memra-engine-tests" --ignored --test-threads=1 tier_transfer::tests:: > "$R/unit/engine-serial.log" 2>&1); rc2=$?
(cd crates/memra-server && CUDA_VISIBLE_DEVICES=0 "$R/bins/ft/memra-server-tests" --ignored --test-threads=1 option_b_ option_c_ > "$R/unit/door-cells.log" 2>&1); rc3=$?
(cd crates/memra-server && "$R/bins/ft/memra-server-tests" > "$R/unit/server-cpu.log" 2>&1); rc4=$?
(cd crates/memra-engine && "$R/bins/ft/memra-engine-tests" tier_transfer:: > "$R/unit/engine-cpu.log" 2>&1); rc5=$?
grep -h '^test result' "$R"/unit/*.log
echo "unit-cells engine-parallel=$rc1 engine-serial=$rc2 door=$rc3 server-cpu=$rc4 engine-cpu=$rc5" | tee -a "$R/unit/run.log"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
