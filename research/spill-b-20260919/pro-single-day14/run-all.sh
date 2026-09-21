#!/usr/bin/env bash
# Day 14 target-card chain: wait for the fix build, check out the gate tip, then base gate (red expected),
# fix gate, serve-smoke and cache-meter on the fix. Every GPU command goes through the collector.
set -uo pipefail
R=/root/spill-receipts/b-day14
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
while tmux has-session -t b14-build-fix 2>/dev/null; do sleep 10; done
cd /root/wt-b
git checkout -q --detach refs/bundle/b14gate || { echo "gate tip checkout failed" >> $R/chain.log; exit 1; }
git rev-parse HEAD > $R/gate-source.txt
bash $R/run-gate.sh main; echo "gate-main rc=$?" >> $R/chain.log
bash $R/run-gate.sh fix; echo "gate-fix rc=$?" >> $R/chain.log
python3 tools/tier-battery.py --rig pro-single --timeout 3000 --out $R/serve-smoke --external-lock --execute bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft > $R/serve-smoke-driver.log 2>&1
echo "serve-smoke rc=$?" >> $R/chain.log; echo $? > $R/serve-smoke.exit
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out $R/cache-meter --external-lock --execute bash $R/cache-meter-cell.sh > $R/cache-meter-driver.log 2>&1
rc=$?; echo "cache-meter rc=$rc" >> $R/chain.log; echo $rc > $R/cache-meter.exit
echo done >> $R/chain.log
