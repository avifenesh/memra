#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign H, the boots a restart
# makes: the first ones after the page cache has been full and filling with no boot going on.
# The sweep of campaign G showed why back-to-back boots hardly fail: after a few boots at the
# floor the kernel keeps several GB free (MemFree rose from 1.5 to 5-11 GB within a dozen boots
# and stayed there), and from then on every boot finds its blocks. So each round here waits for
# the floor again (fill_live.sh: MemFree under 3000 MB, 300 s at most, READERS readers kept
# running) and then makes K boots, arms B and T alternating, GAP_MS apart; rounds repeat until
# UNTIL (a unix time), each in its own cell PREFIX-NN. The first arm alternates between rounds,
# so each arm gets the very first boot of a round equally often.
# usage: campaign_h.sh JOB PREFIX READERS K GAP_MS UNTIL   (from ~/.cache/memra-1316; ends by creating jobs/JOB.done)
set -u
T=research/gb10-boot-oom-20261114/tools
JOB=$1; PREFIX=$2; READERS=$3; K=$4; GAP=$5; UNTIL=$6
i=0
while [ "$(date +%s)" -lt "$UNTIL" ]; do
  i=$((i + 1))
  if [ $((i % 2)) = 1 ]; then arms="B T"; else arms="T B"; fi
  NAME=$PREFIX-$(printf '%02d' $i) READERS=$READERS N=$K GAP_MS=$GAP ARMS="$arms" bash $T/boot_cell.sh
done
touch "jobs/$JOB.done"
