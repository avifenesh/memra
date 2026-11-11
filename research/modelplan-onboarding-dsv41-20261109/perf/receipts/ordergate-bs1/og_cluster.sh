#!/bin/bash
# Dev desk: the order gate with the by-tape interval (bsmma's order_gate.py): bs1 and the router step r1, each
# against k3 (wrx's ordertapes-k1), per group (chat, long) and both together.
set -u
B=$HOME/build/bsmma/og
O=$B/out-cluster; mkdir -p $O
cd $B
PY="nice -n 10 $HOME/.local/bin/uv run --with torch python3 -I $B/order_gate_bt.py --tapes tapes-v2 --ref dec/ref --floor dec/floor"
for arm in bs1 r1; do
  mkdir -p $B/new-cl-$arm
  for g in chat long; do
    for d in tapes-$arm/$g/*/; do ln -sfn $B/$d $B/new-cl-$arm/$(basename $d); done
  done
  ids=$(cat dec/chat-forced.txt dec/long-forced.txt | cut -d' ' -f1 | paste -sd,)
  $PY --cur cur-cl --new new-cl-$arm --cases $ids --json $O/$arm.json > $O/$arm.log 2>&1
  echo "OG $arm rc $?"
done
