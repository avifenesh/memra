#!/usr/bin/env bash
# Design F's native cells (DAY64 section 5 (a)) under the collector's hold: the engine's H2D span cells (the filled batch,
# the plain batch, the fill host function off the owner thread) on the f executable, then the census.
# `tools/tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-f
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
U=$R/unit; mkdir -p "$U"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$U/LOCK.json"
E=$R/bins/f/memra-engine-tests
rc=0
for c in h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies h2d_span_batch_lands_with_its_ticket_on_the_copy_stream h2d_fill_host_function_does_not_hold_the_owner_thread; do
  CUDA_VISIBLE_DEVICES=0 "$E" --include-ignored --exact --nocapture --test-threads=1 tier_transfer::tests::$c > "$U/$c.log" 2>&1; r=$?
  grep -q '^running 1 test$' "$U/$c.log" || { echo "RAN NO TEST" >> "$U/$c.log"; r=97; }
  echo "$(date -u +%FT%TZ) $c rc=$r" | tee -a "$U/run.log"; [ $r = 0 ] || rc=1
done
"$E" day64_ one_side_stream h2d_span_rules > "$U/censuses.log" 2>&1; cr=$?
echo "UNIT native=$rc censuses=$cr" | tee -a "$U/run.log"
[ $rc = 0 ] && [ $cr = 0 ]
