#!/bin/bash
# Independent pass-after for the kv-reuse grid-law fix (memra#1091, PR #1092, binary 9c3a4edd98): the pair cell
# at the RTX 5090 default on the fix binary, the four arms that failed or passed before on main's adee274f0b:
#   compaction off, turn 2 max_tokens 256   (failed before: resume 3113, 373 vs 372)
#   compaction on,  turn 2 max_tokens 96    (failed before: park-compact resume 3113)
#   compaction on,  turn 2 max_tokens 256
#   compaction off, turn 2 max_tokens 96    (control: passed before, resume 3072)
set -u
D=<work>/ab-parkcompact
BIN=$D/memra-server-fix-9c3a4edd98
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
S=$D/status3; echo "start $(date '+%m-%d %H:%M') bin $(sha256sum $BIN | cut -c1-16)" > $S
i=0
for arm in off:256 on:96 on:256 off:96; do
  i=$((i+1)); pc=${arm%%:*}; mt=${arm#*:}
  E="MEMRA_DSPARK_PREFIX_RESTORE=default"; [ $pc = on ] && E="$E MEMRA_KV_PARK_COMPACT=1"
  gpulock run --ttl 30m --for "memra fix stack: pass-after of the kv-reuse grid-law fix (#1092), pair cell compaction $pc MAX_T2=$mt" --wait 2h -- \
    env EV=$D/c3-$i-$pc-mt$mt BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT MAX_T2=$mt WARM_ENV="$E" bash $D/cell/dspark-pair-cell.sh > $D/c3-$i-$pc-mt$mt.log 2>&1
  echo "$i compaction=$pc MAX_T2=$mt rc=$? $(grep -E 'warm==cold' $D/c3-$i-$pc-mt$mt.log | tr '\n' ' ') :: $(grep -a -h -E 'grid-rewind|park-compact|kv-reuse: [0-9]+ of|prefix-cache\] hit' $D/c3-$i-$pc-mt$mt/warm-server.log | cut -c1-130 | tr '\n' '|' | cut -c1-600)" >> $S
done
echo done >> $S
