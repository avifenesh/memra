#!/usr/bin/env bash
# bounded retry: the canonical lock is held by another lane; poll up to 900 s, then run the warm cell through the collector
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
deadline=$((SECONDS + 900))
while ! flock -n /tmp/memra-gpu.lock true; do
  [ $SECONDS -ge $deadline ] && { echo "$(date -u +%FT%TZ) lock still held after 900 s; warm cell not run" >> /root/spill-receipts/b-day26/chain.log; exit 3; }
  sleep 30
done
echo "$(date -u +%FT%TZ) lock free; launching warm cell" >> /root/spill-receipts/b-day26/chain.log
rm -rf /root/spill-receipts/b-day26/cell-warm
python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out /root/spill-receipts/b-day26/cell-warm --execute \
  env WT=/root/wt-b RIGDIR=/root/spill-receipts/b-day26/cells MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18526 WARM_IMMEDIATE=1 \
  bash research/spill-b-20260919/run-day26-cell.sh warm AB > /root/spill-receipts/b-day26/collector-warm.log 2>&1
echo "warm exit=$?" >> /root/spill-receipts/b-day26/chain.log
