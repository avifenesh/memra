#!/bin/bash
# Dev desk: the order gate for ONE tape group of bs1 against k3 (the current route = wrx's ordertapes-k1), with the
# route log's router table. usage: og_group.sh GROUP
set -u
g=$1
B=$HOME/build/bsmma/og
O=$B/out; mkdir -p $O
cd $B
ids=$(cut -d' ' -f1 dec/$g-forced.txt | paste -sd,)
nice -n 10 /home/avifenesh/.local/bin/uv run --with torch python3 -I $B/order_gate.py --tapes tapes-v2 --ref dec/ref --floor dec/floor \
  --cur receipts-perf/ordertapes-k1/$g --new tapes-bs1/$g --cases $ids \
  --routes-cur receipts-perf/ordertapes-k1/$g --routes-new tapes-bs1/$g --json $O/$g.json > $O/$g.log 2>&1
echo "OG $g rc $?"
