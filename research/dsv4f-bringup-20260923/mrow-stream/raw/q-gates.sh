#!/usr/bin/env bash
# q-gates.sh: DSpark identity gates with the gate's required env (q-fin and q-mrow ran them without
# MEMRA_DSV4_DRAFTER / MEMRA_DSV4_DECODE_PATH and got REFUSE rc=2). Fin tree served + historical,
# then the mrow lane tree served (lane read unset, so the shipped dispatch).
while ! grep -q MROW_DONE /root/rcpt/q-mrow.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-gates.summary
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
unset MEMRA_DSV4_MOE_MROW_STREAM MEMRA_DSV4_MOE_M1_STREAM
F=/root/lane/target-fin/release; X=/root/lane/target-mrow/release; G=/root/rcpt/gates
wait_lock; /root/box/gate.sh $G/fin-served $F/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $G/fin-served/out 2 0,1 --served
echo "fin-served $(grep -h GATE $G/fin-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
wait_lock; /root/box/gate.sh $G/fin-hist $F/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $G/fin-hist/out 2 0,1
echo "fin-hist $(grep -h GATE $G/fin-hist/gate.log | tail -2 | tr '\n' ' ')" >> $S
wait_lock; /root/box/gate.sh $G/mrow-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $G/mrow-served/out 2 0,1 --served
echo "mrow-served $(grep -h GATE $G/mrow-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
echo GATES_DONE >> $S
