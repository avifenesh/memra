#!/usr/bin/env bash
# q-m1ab.sh: MEMRA_DSV4_MOE_M1_STREAM served A/B on one binary, one boot per row,
# order A B B A A B B A A B (A = stream ON), then the TP/EP replay ON OFF OFF ON.
while ! grep -q Q_DONE /root/rcpt/q-rebase.summary 2>/dev/null; do sleep 20; done
S=/root/rcpt/q-m1ab.summary
X=/root/lane/target-m1/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
i=0
for arm in on off off on on off off on on off; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_DSV4_MOE_M1_STREAM=1 MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_MOE_M1_STREAM=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh /root/rcpt/m1ab/r$i-$arm $X/memra-server /root/box/ab-cells.txt "$@" > /root/rcpt/m1ab/r$i-$arm.out 2>&1
  echo "r$i $arm rc=$? $(grep 'CELL greedy-c1' /root/rcpt/m1ab/r$i-$arm/controller.log)" >> $S
done
j=0
for arm in on off off on; do
  j=$((j+1)); wait_lock
  ( export MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference MEMRA_DSV4_ATTENTION_TP_GATE=1 MEMRA_DSV4_SAMPLER=device
    if [[ $arm == on ]]; then export MEMRA_DSV4_MOE_M1_STREAM=1; fi
    /root/box/gate.sh /root/rcpt/m1ab/tpep-t$j-$arm $X/dsv4_tp_ep_sampled_perf_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt --full-token-replay )
  echo "tpep t$j $arm $(grep REPLAY_SUMMARY /root/rcpt/m1ab/tpep-t$j-$arm/gate.log | cut -c1-160) $(tail -1 /root/rcpt/m1ab/tpep-t$j-$arm/gate.log)" >> $S
done
echo AB_DONE >> $S
