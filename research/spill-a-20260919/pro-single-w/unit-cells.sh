#!/usr/bin/env bash
# W's identity cells (DAY61 section 1 (a)) under the collector's hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash unit-cells.sh @COLLECTOR_LOCK_FD@`.
# The test executables were built by build.sh outside the hold. Green arm: the w executables must pass the host and the
# pinned (both arms) identity cells. Red arm: the red executables (the streamed copy skips each chunk's second vector,
# its marker printed) must fail both. Then the CPU census.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-w
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
U=$R/unit; mkdir -p "$U"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$U/LOCK.json"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$U/compute-apps.before.csv" 2>&1
cell() { # name exe filter
  local name=$1 exe=$2 filter=$3
  CUDA_VISIBLE_DEVICES=0 "$exe" --ignored --exact --nocapture --test-threads=1 "$filter" > "$U/$name.log" 2>&1
  local rc=$?; echo "$rc" > "$U/$name.exit"; echo "$(date -u +%FT%TZ) $name rc=$rc $(grep -h '^test result' "$U/$name.log")" | tee -a "$U/run.log"
}
cell a1-green "$R/bins/w/memra-engine-tests" tier_transfer::tests::day61_the_streamed_checksum_is_the_checksum
cell a2-green "$R/bins/w/memra-engine-tests" tier_transfer::tests::day61_the_streamed_checksum_is_the_checksum_on_pinned_memory
cell a1-red "$R/bins/red/memra-engine-tests" tier_transfer::tests::day61_the_streamed_checksum_is_the_checksum
cell a2-red "$R/bins/red/memra-engine-tests" tier_transfer::tests::day61_the_streamed_checksum_is_the_checksum_on_pinned_memory
"$R/bins/w/memra-engine-tests" day61_the_pinned_views native_cells_own > "$U/censuses.log" 2>&1; echo "$?" > "$U/censuses.exit"
echo "$(date -u +%FT%TZ) censuses rc=$(cat "$U/censuses.exit") $(grep -h '^test result' "$U/censuses.log")" | tee -a "$U/run.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$U/compute-apps.after.csv" 2>&1
g1=$(cat "$U/a1-green.exit"); g2=$(cat "$U/a2-green.exit"); r1=$(cat "$U/a1-red.exit"); r2=$(cat "$U/a2-red.exit")
m1=$(grep -c 'day61 red arm' "$U/a1-red.log"); m2=$(grep -c 'day61 red arm' "$U/a2-red.log")
echo "UNIT a1-green=$g1 a2-green=$g2 a1-red=$r1 (marker $m1) a2-red=$r2 (marker $m2) censuses=$(cat "$U/censuses.exit")" | tee -a "$U/run.log"
[ "$g1" = 0 ] && [ "$g2" = 0 ] && [ "$r1" != 0 ] && [ "$r2" != 0 ] && [ "$m1" -gt 0 ] && [ "$m2" -gt 0 ] && [ "$(cat "$U/censuses.exit")" = 0 ]
