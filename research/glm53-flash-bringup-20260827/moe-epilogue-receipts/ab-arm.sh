#!/bin/bash
# One BOOT of the interleaved A/B for MEMRA_MOE_FUSED_EPI, at the SERVING slot count.
# usage: ab-arm.sh <tag> <0|1>
#
# Composed only of pieces already exercised on this box: cell-epi.sh (PID-verified stop),
# probe.py and steady.py (the banked decode-attribution instrument, unchanged except for the
# engagement fields steady.py now reports).
#
# BASE ENV = A4BEST, the best single-card config the attribution measured
# (ROADMAP.txt step 2b: ST_PINNED + BF16_MMV + 14000 slots, 25.9 tok/s greedy AND sampled).
# The A/B changes ONE variable against it. 14000 slots is also the point of the exercise for
# this lane's finding 3: at a three-slot margin the fused arm staged 1.19 MB/token MORE because
# it admits a token's whole working set before running anything. At 14000 slots against a
# 24-block working set the margin is ~583x, and steady.py's MB_per_tok / miss_per_tok medians
# are the receipt for whether that cost survives into the serving regime or vanishes.
#
# THE INTERLEAVE UNIT IS A BOOT, not a request: these are process-level env reads behind
# OnceLock-style gates, so they cannot alternate inside one server. Same deviation the
# attribution documented (METHOD.txt), stated rather than hidden. The caller alternates
# tags A0/A1/B0/B1/... x5 and never runs the two arms back to back from one boot.
#
# IDLE IS CHECKED PER BOOT, not once per session. The box is shared with the PP lane and two
# lanes timing on one box invalidates both.
set -u
TAG=$1; EPI=$2

~/idle-check.sh || { echo "ABORT $TAG: box not idle — a timing arm must not start."; exit 2; }

~/cell-epi.sh "$TAG" MEMRA_ST_PINNED=1 MEMRA_BF16_MMV=1 MEMRA_MOE_RESIDENT=0 \
    MEMRA_MOE_SLOTS=14000 MEMRA_MOE_FUSED_EPI="$EPI" || exit 1

# Warm the expert pool across several prompts so the steady rows are not measuring first-touch.
for i in 0 2 5 7 9; do python3 ~/probe.py warm-$i greedy 24 $i > /dev/null 2>&1; done

# greedy = the INSTRUMENT (byte-deterministic, gives the identity oracle).
python3 ~/steady.py ~/cell-$TAG.log $TAG-p5-greedy  greedy  5 192 4
# vendor-default sampled = the PRODUCT. A default flip is justified by THIS row, never by the
# greedy one (LAW:never-serve-greedy-verify-sampled).
python3 ~/steady.py ~/cell-$TAG.log $TAG-p5-sampled sampled 5 192 4
python3 ~/steady.py ~/cell-$TAG.log $TAG-p7-greedy  greedy  7 192 4

echo "### $TAG engagement (full engagement on glm5_next = 42.0 dispatches/token)"
grep -E "\[moe-fused-epi\] snapshot" ~/cell-$TAG.log | tail -1
grep -E "\[moe-cache\] snapshot" ~/cell-$TAG.log | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "ARMDONE-$TAG"
