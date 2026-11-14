#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign B, driver-level boots timed
# to the moment the readers have used up what an exiting process freed.
#   b1  page cache full, two readers live, 10 times: a 16 GiB holder exits, then for 25 s armed
#       boots, each creating its context when MemFree falls under 800 MB
# usage: campaign_b.sh   (from ~/.cache/memra-1316; ends by creating jobs/campaign-b.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=b1-live2-dip STATE=live MODE=dip TRIALS=10 HOLD_GIB=16 HOLD_S=2 BURST=25 ARM_MB=800 READERS=2 RETRY_MAX=40 bash $T/probe_cell.sh
touch jobs/campaign-b.done
