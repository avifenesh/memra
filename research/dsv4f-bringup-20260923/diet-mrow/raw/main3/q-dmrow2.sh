#!/usr/bin/env bash
# q-dmrow2.sh (dev pair): redo of the S1 served A/B. The first run (q-dev2.sh) is void: its A/B
# patch did not apply, so both arms were the lane binary. Here the arms are two binaries: lane =
# lane/dsv4-diet-mrow-20260923 58348d94d (main plus b7e837993), base = main 7029cd67c (the lat2
# base build). DSpark order l b b l l b b l, then plain l b b l l b b l, one boot per row.
while ! grep -q LAT2_DONE /root/rcpt/q-lat2.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-dmrow2.summary; R=/root/rcpt/dmrow2; mkdir -p $R/spec $R/plain
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
L=/root/lane/target-dmrow/release/memra-server
B=/root/lane/target-latbase/release/memra-server
echo "lane $(sha256sum $L | cut -c1-16) base $(sha256sum $B | cut -c1-16) base-tree $(git -C /root/lane/memra-latbase rev-parse HEAD)" >> $S
i=0
for arm in lane base base lane lane base base lane; do
  i=$((i+1)); wait_lock; d=$B; [[ $arm == lane ]] && d=$L
  /root/box/cell.sh $R/spec/s$i-$arm $d /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark MEMRA_ENV_AUDIT=warn > $R/spec/s$i-$arm.out 2>&1
  echo "spec s$i $arm rc=$? $(grep -hE 'CELL ' $R/spec/s$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
i=0
for arm in lane base base lane lane base base lane; do
  i=$((i+1)); wait_lock; d=$B; [[ $arm == lane ]] && d=$L
  /root/box/cell.sh $R/plain/r$i-$arm $d /root/box/cells-spec.txt MEMRA_ENV_AUDIT=warn > $R/plain/r$i-$arm.out 2>&1
  echo "plain r$i $arm rc=$? $(grep -hE 'CELL ' $R/plain/r$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo DMROW2_DONE >> $S
