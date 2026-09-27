#!/usr/bin/env bash
# Day 13 serving batteries on the fix, queued behind the final gate pair; collector holds the lock.
set -uo pipefail
R=/root/spill-receipts/b-day13
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
while tmux has-session -t b13-finals 2>/dev/null; do sleep 15; done
cd /root/wt-b
git rev-parse HEAD > $R/batteries-source.txt
python3 tools/tier-battery.py --rig pro-single --timeout 3000 --out $R/serve-smoke --execute bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft > $R/serve-smoke-driver.log 2>&1
echo $? > $R/serve-smoke.exit
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out $R/cache-meter --execute bash $R/cache-meter-cell.sh > $R/cache-meter-driver.log 2>&1
echo $? > $R/cache-meter.exit
