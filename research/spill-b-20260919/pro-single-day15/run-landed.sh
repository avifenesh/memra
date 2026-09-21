#!/usr/bin/env bash
# Day 15 landed-binary battery: build memra-server at the landing ref, then the day-14 twin gate
# (must stay PASS on the new default), serve-smoke and cache-meter through the collector, then the
# collector's --validate over every cell of the day. Every GPU command goes through the collector.
set -uo pipefail
R=/root/spill-receipts/b-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
bash $R/build-arm.sh landed refs/bundle/b15landed; echo "build-landed rc=$?" >> $R/landed.log
[ "$(cat $R/build-landed/exit)" = 0 ] || { echo "build-landed failed; chain stops" >> $R/landed.log; exit 1; }
git rev-parse HEAD > $R/gate-source-landed.txt
python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out $R/gate-landed --external-lock \
  --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
    --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --bin $R/bins/landed/memra-server \
    --out $R/gate-landed/cell --budget-mib 1024 --cohort-tokens 2800,3000,3200 \
    --turns 8 --start-tokens 9200 --grow-tokens 300 > $R/gate-landed-driver.log 2>&1
rc=$?; echo "gate-landed rc=$rc" >> $R/landed.log; echo $rc > $R/gate-landed.exit
# serve-smoke and cache-meter carry no @COLLECTOR_LOCK_FD@ argument: the collector holds the lock.
python3 tools/tier-battery.py --rig pro-single --timeout 3000 --out $R/serve-smoke --execute bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft > $R/serve-smoke-driver.log 2>&1
rc=$?; echo "serve-smoke rc=$rc" >> $R/landed.log; echo $rc > $R/serve-smoke.exit
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out $R/cache-meter --execute bash $R/cache-meter-cell.sh > $R/cache-meter-driver.log 2>&1
rc=$?; echo "cache-meter rc=$rc" >> $R/landed.log; echo $rc > $R/cache-meter.exit
for c in ab-smoke ab-full gate-landed serve-smoke cache-meter; do
  [ -d $R/$c ] && python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/landed.log
done
echo done >> $R/landed.log
