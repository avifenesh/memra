#!/usr/bin/env bash
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
for o in AB BA; do
  lo=$(echo $o | tr A-Z a-z)
  python3 tools/tier-battery.py --rig pro-single --timeout 5400 --out /root/spill-receipts/b-day26/cell-$lo --execute \
    env WT=/root/wt-b RIGDIR=/root/spill-receipts/b-day26/cells MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18526 \
    bash research/spill-b-20260919/run-day26-cell.sh $lo $o > /root/spill-receipts/b-day26/collector-$lo.log 2>&1
  echo "$o exit=$?" >> /root/spill-receipts/b-day26/chain.log
done
