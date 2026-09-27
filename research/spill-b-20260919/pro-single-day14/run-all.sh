#!/usr/bin/env bash
# Day 14 target-card chain, second sitting: build both arms at their exact refs (base = origin/main,
# fix = the lane tip), check out the gate tip, then base gate (red expected), fix gate, serve-smoke
# and cache-meter on the fix, then the collector's --validate over every cell. Every GPU command
# goes through the collector; builds and validation take no lock.
set -uo pipefail
R=/root/spill-receipts/b-day14
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
bash $R/build-arm.sh main refs/bundle/b14base; echo "build-main rc=$?" >> $R/chain.log
bash $R/build-arm.sh fix refs/bundle/b14tip; echo "build-fix rc=$?" >> $R/chain.log
for a in main fix; do [ "$(cat $R/build-$a/exit)" = 0 ] || { echo "build-$a failed; chain stops" >> $R/chain.log; exit 1; }; done
git checkout -q --detach refs/bundle/b14tip || { echo "gate tip checkout failed" >> $R/chain.log; exit 1; }
git rev-parse HEAD > $R/gate-source.txt
bash $R/run-gate.sh main; echo "gate-main rc=$?" >> $R/chain.log
bash $R/run-gate.sh fix; echo "gate-fix rc=$?" >> $R/chain.log
# serve-smoke and cache-meter carry no @COLLECTOR_LOCK_FD@ argument: the collector holds the lock
# for them (day-13 shape); --external-lock would be refused (the first sitting's second refusal).
python3 tools/tier-battery.py --rig pro-single --timeout 3000 --out $R/serve-smoke --execute bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft > $R/serve-smoke-driver.log 2>&1
rc=$?; echo "serve-smoke rc=$rc" >> $R/chain.log; echo $rc > $R/serve-smoke.exit
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out $R/cache-meter --execute bash $R/cache-meter-cell.sh > $R/cache-meter-driver.log 2>&1
rc=$?; echo "cache-meter rc=$rc" >> $R/chain.log; echo $rc > $R/cache-meter.exit
for c in gate-main gate-fix serve-smoke cache-meter; do
  [ -d $R/$c ] && python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/chain.log
done
echo done >> $R/chain.log
