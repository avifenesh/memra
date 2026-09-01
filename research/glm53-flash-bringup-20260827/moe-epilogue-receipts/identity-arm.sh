#!/bin/bash
# Real-width bit-identity spot-check for one arm of MEMRA_MOE_FUSED_EPI, on the 178 GB artifact.
# CORRECTNESS ONLY. No number this produces is a timing claim: the box has a co-tenant lane.
#
# Pinned to CARD 1. The co-tenant peaked at 82 GB on card 0; card 1 has sat at ~650 MiB all day.
# Same card class either way, so an identity comparison with both arms on card 1 is valid.
#
# EXCLUSIVITY RE-CHECK, as a grep and not a judgement call: the other lane declares an exclusive
# window in its own command line. If any live process declares one, this script refuses. That is
# what stood this lane down at 14:32 and it must not depend on somebody remembering to look.
set -u
ARMTAG=$1; EPI=$2
# Walk /proc directly instead of grepping ps. A `ps | grep PATTERN` pipeline matches ITS OWN
# command line, because the pattern is literally in it — this guard aborted twice on nothing but
# itself before the check was written this way (the same self-match that had to be fixed in
# idle-check.sh an hour earlier; twice in one session is a pattern, not an accident).
# A candidate counts only if its cmdline declares exclusivity AND is not a grep/ps/this script.
DECLARED=""
for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    cl=$(tr "\0" " " < "$d/cmdline" 2>/dev/null) || continue
    case "$cl" in
        *identity-arm*|*grep*|*" ps "*|"ps "*) continue;;
    esac
    case "$cl" in
        *[Ee]xclusive\ box*) DECLARED="$DECLARED $pid";;
    esac
done
if [ -n "${DECLARED// /}" ]; then
    echo "ABORT $ARMTAG: another lane declares an exclusive box window."
    for pid in $DECLARED; do echo "  pid=$pid $(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null | head -c 140)"; done
    exit 2
fi
~/cell-epi.sh $ARMTAG CUDA_VISIBLE_DEVICES=1 MEMRA_ST_PINNED=1 MEMRA_BF16_MMV=1 \
    MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=1000 MEMRA_MOE_FUSED_EPI=$EPI || exit 1
for i in 0 5; do python3 ~/probe.py warm-$i greedy 16 $i > /dev/null 2>&1; done
echo "### $ARMTAG greedy p5"
python3 ~/probe.py $ARMTAG-p5-greedy greedy 96 5
echo "### $ARMTAG sampled p5 (vendor default, seeded)"
python3 ~/probe.py $ARMTAG-p5-sampled sampled 96 5
echo "### $ARMTAG greedy p7"
python3 ~/probe.py $ARMTAG-p7-greedy greedy 96 7
echo "### $ARMTAG engagement (full = 42.0 dispatches/token)"
grep -E "\[moe-fused-epi\] snapshot" ~/cell-$ARMTAG.log | tail -1
grep -E "\[moe-cache\] snapshot" ~/cell-$ARMTAG.log | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ARMDONE-$ARMTAG
