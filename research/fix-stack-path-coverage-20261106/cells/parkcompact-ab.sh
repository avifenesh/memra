#!/bin/bash
# A/B on main's own binary (adee274f0b = main 05a8a63503 + #1085's prime-fairness fix, which touches neither park
# compaction nor the restore paths): the DSpark pair cell at the RTX 5090 default (restore switch OFF), with
# main's MEMRA_KV_PARK_COMPACT door off and on. Question: does park compaction alone break warm==cold on turn 3
# (the stack's #1062 makes it the 5090 default, and the stack's default-switch cell failed 373 vs 372)?
# Order: off, on, on, off (both orders).
set -u
W=<work>/engine-dsv4-glm
D=<work>/ab-parkcompact
BIN=$D/memra-server-main-adee274f0b
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
S=$D/status; echo "start $(date '+%m-%d %H:%M') bin $(sha256sum $BIN | cut -c1-16)" > $S
cd $W || exit 2
i=0
for arm in off on on off; do
  i=$((i+1))
  if [ $arm = on ]; then E="MEMRA_DSPARK_PREFIX_RESTORE=default MEMRA_KV_PARK_COMPACT=1"; else E="MEMRA_DSPARK_PREFIX_RESTORE=default"; fi
  gpulock run --ttl 45m --for "memra fix stack: park-compact A/B on main's binary, pair cell at the 5090 default ($arm)" --wait 2h -- \
    env EV=$D/cell-$i-$arm BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="$E" bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $D/cell-$i-$arm.log 2>&1
  echo "$i $arm rc=$? $(grep -E 'warm==cold|DSPARK-PAIR CELL' $D/cell-$i-$arm.log | tr '\n' ' ') :: $(grep -a -h -E 'park-compact|kv-reuse: [0-9]+ of|prefix-cache\] hit|restore grid law' $D/cell-$i-$arm/warm-server.log | cut -c1-140 | tr '\n' '|' | cut -c1-700)" >> $S
done
echo done >> $S
