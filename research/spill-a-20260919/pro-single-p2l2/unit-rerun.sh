#!/usr/bin/env bash
# DAY67 section 4: P2L2's (a) unit step rerun whole on P2's tree with the corrected worker cell arithmetic (branch
# lane/spill-a-p2l2-unit-20260926 at a2419d3e1 = 064f9fa0d + the test-only fixes 411177fea and 28c7aa6c1; the production
# code is 064f9fa0d's, byte for byte). Its own clone (/root/wt-a-p2unit), so no running sitting's tree is touched. Step
# 1 builds the test executables outside any hold; step 2 is the same unit cells as unit-cells.sh, under ONE collector
# hold: `bash unit-rerun.sh build`, then
# `tools/tier-battery.py --rig pro-single --external-lock --execute bash unit-rerun.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
R=/root/spill-receipts/a-p2l2/unit-rerun
W=/root/wt-a-p2unit
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R"
if [ "${1:-}" = build ]; then
  [ -d "$W/.git" ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git "$W" >> "$R/build.log" 2>&1
  cd "$W" || exit 1
  git fetch -q origin lane/spill-a-p2l2-unit-20260926 >> "$R/build.log" 2>&1
  git checkout -q -B p2unit a2419d3e1 >> "$R/build.log" 2>&1 || { echo "rc=2 (checkout)" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree.sha"
  for pkg in "memra-engine --lib" "memra-server --lib" "memra-tier --test contracts"; do
    nice -n 5 cargo test -p $pkg --no-run >> "$R/build.log" 2>&1 || { echo "rc=1 ($pkg)" >> "$R/build.log"; exit 1; }
  done
  echo "rc=0" >> "$R/build.log"
  exit 0
fi
fd=$1
cd "$W" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/LOCK.json"
green=0
for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --nocapture tier_transfer::tests:: > "$R/engine-parallel-$r.log" 2>&1
  rc=$?; [ $rc -eq 0 ] && green=$((green + 1))
  echo "parallel run $r rc=$rc $(grep -h '^test result' "$R/engine-parallel-$r.log")" | tee -a "$R/run.log"
done
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-engine --lib -- --ignored --test-threads=1 --nocapture tier_transfer::tests:: > "$R/engine-serial.log" 2>&1
rc2=$?
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/door-cells.log" 2>&1
rc=$?; grep -E '^test |^test result' "$R/door-cells.log"
cargo test -p memra-server --lib -- hash_ hashing_ every_path_that_meets_a_hashing every_path_that_meets_a_demoting every_path_that_meets_a_promoting the_run_loop_waits_boundedly the_d2h_spans day31_ day33_ day34_ day35_ day36_ day38_ day41_ day42_ day47_ day49_ day51_ day52_ pause_ > "$R/cpu-censuses.log" 2>&1
rc3=$?
cargo test -p memra-engine --lib -- rules_are_as_stated native_cells_own_their_context one_side_stream_beside_the_owner span_receipt_rules day39_ > "$R/engine-censuses.log" 2>&1
rc4=$?
cargo test -p memra-tier --test contracts -- d2h_span h2d_ d2h_device_receipt d2d_ span_receipt > "$R/tier.log" 2>&1
rc5=$?
echo "unit-cells parallel=$green/3 engine-serial-rc=$rc2 door-rc=$rc cpu-rc=$rc3 engine-census-rc=$rc4 tier-rc=$rc5" | tee -a "$R/run.log"
[ $green -eq 3 ] && [ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
