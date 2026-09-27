#!/usr/bin/env bash
# q-m1spec.sh: served DSpark with the M1 stream ON vs OFF (order on off off on, one boot per row),
# then an nsys profile of served plain with the stream ON, so the lever ranking starts on top of it.
while ! grep -q PROF_Q_DONE /root/rcpt/q-prof.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-m1spec.summary
X=/root/lane/target-m1/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
mkdir -p /root/rcpt/m1spec
i=0
for arm in on off off on; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then v=1; else v=0; fi
  /root/box/cell.sh /root/rcpt/m1spec/s$i-$arm $X/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_MOE_M1_STREAM=$v MEMRA_ENV_AUDIT=warn > /root/rcpt/m1spec/s$i-$arm.out 2>&1
  echo "s$i $arm rc=$? $(grep 'CELL ' /root/rcpt/m1spec/s$i-$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
done
wait_lock; /root/box/prof-served.sh /root/rcpt/prof/served-plain-m1 $X/memra-server MEMRA_DSV4_MOE_M1_STREAM=1 MEMRA_ENV_AUDIT=warn > /root/rcpt/prof/served-plain-m1.out 2>&1
echo "served-plain-m1 rc=$? $(tail -1 /root/rcpt/prof/served-plain-m1/controller.log)" >> $S
echo SPEC_Q_DONE >> $S
