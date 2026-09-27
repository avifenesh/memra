#!/usr/bin/env bash
# w-skip.sh: the TP/EP rows in q-m1ab ran the lane binary whose down order put the stream ahead of
# the gate-armed half2 tail; the shipped order keeps the gate arm first, so those rows measure a
# program that no longer exists. Stop q-m1ab after r10 and release the queue.
S=/root/rcpt/q-m1ab.summary
while ! grep -q '^r10 ' $S 2>/dev/null; do sleep 15; done
for p in $(pgrep -f 'box/q-m1ab.sh'); do kill $p; done
sleep 2
pkill -f dsv4_tp_ep_sampled_perf_gate; pkill -f 'box/gate.sh'
sleep 5
echo "tpep rows skipped: the lane binary's stream-first down order was superseded (gate-armed tails keep precedence)" >> $S
echo AB_DONE >> $S
