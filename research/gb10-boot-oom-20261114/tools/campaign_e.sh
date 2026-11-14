#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign E, driver-level boots spaced
# GAP_MS apart with the page cache full and READERS readers live. Campaigns A and D booted (or
# created contexts) back to back, so each one found the blocks the one before had just freed; here
# the readers have GAP_MS to take them first, as they have before a real restart.
# usage: campaign_e.sh NAME N GAP_MS READERS   (from ~/.cache/memra-1316; ends by creating jobs/NAME.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=$1 STATE=live MODE=loop N=$2 GAP_MS=$3 READERS=$4 RETRY_MAX=200 bash $T/probe_cell.sh
touch "jobs/$1.done"
