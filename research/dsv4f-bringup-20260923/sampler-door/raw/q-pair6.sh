#!/usr/bin/env bash
# q-pair6.sh (2x RTX PRO 6000 WS pod), after q-pair5.sh: the MEMRA_DSV4_SAMPLER door (decide-by
# 2026-09-21, passed). Plain served cells on main 25bbb91f5, host vs device sampler, one boot
# per row, order h d d h h d d h h d. Greedy never samples, so its hashes must match across arms;
# sampled hashes are compared within each arm (determinism) and across arms (class change).
while ! grep -q PAIR5_DONE /root/rcpt/q-pair5.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair6.summary; R=/root/rcpt/sampler; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
B=/root/lane/target-main5/release/memra-server
echo "base $(sha256sum $B | cut -c1-16)" >> $S
i=0
for arm in h d d h h d d h h d; do
  i=$((i+1)); wait_lock; d=$R/r$i-$arm
  env_arm=""; [[ $arm == d ]] && env_arm="MEMRA_DSV4_SAMPLER=device"
  # shellcheck disable=SC2086
  /root/box/cell.sh $d $B /root/box/cells-spec.txt $env_arm > $d.out 2>&1
  echo "sampler r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
echo PAIR6_DONE >> $S
