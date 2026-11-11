#!/bin/bash
# Dev desk: the order gate over the chat and long groups together (bs1 against k3), no router table.
set -u
B=$HOME/build/bsmma/og
O=$B/out
mkdir -p $B/cur-cl $B/new-cl
cd $B
for g in chat long; do
  for d in receipts-perf/ordertapes-k1/$g/*/; do ln -sfn $B/$d $B/cur-cl/$(basename $d); done
  for d in tapes-bs1/$g/*/; do ln -sfn $B/$d $B/new-cl/$(basename $d); done
done
ids=$(cat dec/chat-forced.txt dec/long-forced.txt | cut -d' ' -f1 | paste -sd,)
nice -n 10 $HOME/.local/bin/uv run --with torch python3 -I $B/order_gate.py --tapes tapes-v2 --ref dec/ref \
  --floor dec/floor --cur cur-cl --new new-cl --cases $ids --json $O/chat+long.json > $O/chat+long.log 2>&1
echo "OG chat+long rc $?"
