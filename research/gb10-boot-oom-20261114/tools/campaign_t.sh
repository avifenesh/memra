#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign T, restarts right after a
# process that held device memory exits, with the page cache full and readers live
# (transition_cell.sh), arms B (the red arm) and T (the head) alternating.
# usage: campaign_t.sh JOB NAME READERS K GAP_MS HOLD_GIB ARM_MB UNTIL   (from ~/.cache/memra-1316; ends by creating jobs/JOB.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=$2 READERS=$3 K=$4 GAP_MS=$5 ARMS="B T" HOLD_GIB=$6 ARM_MB=$7 UNTIL=$8 bash $T/transition_cell.sh
touch "jobs/$1.done"
