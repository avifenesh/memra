#!/bin/bash
# Is the off-grid resume main's own, with no park compaction? Main's binary (adee274f0b), the pair cell at the
# 5090 default (switch OFF), MEMRA_KV_PARK_COMPACT unset, turn 2 asked with MAX_T2=256 so the parked turn-2
# session's cap covers turn 3's (plain_resume_cap_admits: entry_cap >= ctx_cap). Turn 2's greedy reply ends at
# its EOS either way. Twice. Then a control at MAX_T2=96 once more (expected warm==cold, resume 3072).
set -u
D=<work>/ab-parkcompact
BIN=$D/memra-server-main-adee274f0b
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
S=$D/status2; echo "start $(date '+%m-%d %H:%M') bin $(sha256sum $BIN | cut -c1-16)" > $S
i=0
for mt in 256 256 96; do
  i=$((i+1))
  gpulock run --ttl 45m --for "memra fix stack: main's plain resume without park compaction, pair cell MAX_T2=$mt" --wait 2h -- \
    env EV=$D/c2-$i-mt$mt BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT MAX_T2=$mt WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=default" bash $D/cell/dspark-pair-cell.sh > $D/c2-$i-mt$mt.log 2>&1
  echo "$i MAX_T2=$mt rc=$? $(grep -E 'warm==cold' $D/c2-$i-mt$mt.log | tr '\n' ' ') :: $(grep -a -h -E 'park-compact|kv-reuse: [0-9]+ of|prefix-cache\] hit' $D/c2-$i-mt$mt/warm-server.log | cut -c1-120 | tr '\n' '|' | cut -c1-500)" >> $S
done
echo done >> $S
