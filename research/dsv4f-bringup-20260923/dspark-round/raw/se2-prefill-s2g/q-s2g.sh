#!/usr/bin/env bash
# q-s2g.sh (second SE pair): the DSpark round lane moves the chunked prefill's row placement too,
# so plain TTFT, snap binary (C) against the round lane (E), PDL and vocab head on, C E E C C E.
set -u
S=/root/rcpt/q-s2g.summary; R=/root/rcpt/prefill-rows-s2g; mkdir -p $R
until grep -q S2F_DONE /root/rcpt/q-s2f.summary 2>/dev/null; do sleep 60; done
XC=/root/lane/target-snap/release; XE=/root/lane/target-dsr/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { case $1 in C) echo $XC ;; *) echo $XE ;; esac; }
i=0
for arm in C E E C C E; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $arm)/memra-server /root/box/cells-ttft.txt MEMRA_ENV_AUDIT=on MEMRA_DSV4_PDL=1 MEMRA_DSV4_VOCAB_HEAD=1 > $d.out 2>&1
  echo "serve r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo S2G_DONE >> $S
