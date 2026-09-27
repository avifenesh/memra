#!/usr/bin/env bash
# q-rebase.sh: re-anchor on the healthy pair. main binaries first (TP/EP replay, served plain,
while [[ ! -f /root/buildall.done ]]; do sleep 20; done
# served DSpark), then the M1-stream served identity gate ON and OFF. One campaign at a time.
S=/root/rcpt/q-rebase.summary
M=/root/lane/target-main/release; X=/root/lane/target-m1/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
wait_lock
( export MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference MEMRA_DSV4_ATTENTION_TP_GATE=1 MEMRA_DSV4_SAMPLER=device
  /root/box/gate.sh /root/rcpt/tpep-replay-main-r1 $M/dsv4_tp_ep_sampled_perf_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt --full-token-replay )
echo "tpep-main $(grep REPLAY_SUMMARY /root/rcpt/tpep-replay-main-r1/gate.log | cut -c1-200) $(tail -1 /root/rcpt/tpep-replay-main-r1/gate.log)" >> $S
wait_lock
/root/box/cell.sh /root/rcpt/base-plain-r1 $M/memra-server /root/box/cells-base.txt > /root/rcpt/base-plain-r1.out 2>&1
echo "PLAIN rc=$?" >> $S
wait_lock
/root/box/cell.sh /root/rcpt/spec-dspark-r1 $M/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark > /root/rcpt/spec-dspark-r1.out 2>&1
echo "SPEC rc=$?" >> $S
for arm in on off; do
  wait_lock
  if [[ $arm == on ]]; then export MEMRA_DSV4_MOE_M1_STREAM=1; else unset MEMRA_DSV4_MOE_M1_STREAM; fi
  MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_DRAFTER=dspark /root/box/gate.sh /root/rcpt/m1-dspark-served-$arm $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json /root/rcpt/m1-dspark-served-$arm/out 2 0,1 --served
  echo "m1-$arm $(grep -h 'GATE' /root/rcpt/m1-dspark-served-$arm/gate.log | tail -2 | tr '\n' ' ')" >> $S
done
unset MEMRA_DSV4_MOE_M1_STREAM
echo Q_DONE >> $S
