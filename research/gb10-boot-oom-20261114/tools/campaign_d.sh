#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign D, the allocation a boot
# needs sampled about five times a second (one process creating and destroying contexts), to find
# the host state in which it fails.
#   d0  the host as found (page cache full from the cells before, no reader), 40 s
#   d1  page cache full, two readers live, 240 s
#   d2  page cache full, four readers live, 240 s
#   d3  page cache full, two readers live, a 16 GiB holder exiting again and again, 480 s
#   d4  page cache full, eight readers live, 240 s
# A failed creation is retried in place, 50 ms apart, up to 200 times (10 s).
# usage: campaign_d.sh   (from ~/.cache/memra-1316; ends by creating jobs/campaign-d.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=d0-asis-churn STATE=asis MODE=churn SECONDS_RUN=40 HOLD_GIB=0 RETRY_MAX=200 bash $T/probe_cell.sh
NAME=d1-live2-churn STATE=live MODE=churn SECONDS_RUN=240 HOLD_GIB=0 READERS=2 RETRY_MAX=200 bash $T/probe_cell.sh
NAME=d2-live4-churn STATE=live MODE=churn SECONDS_RUN=240 HOLD_GIB=0 READERS=4 RETRY_MAX=200 bash $T/probe_cell.sh
NAME=d3-live2-hold-churn STATE=live MODE=churn SECONDS_RUN=480 HOLD_GIB=16 HOLD_S=5 HOLD_GAP_S=20 READERS=2 RETRY_MAX=200 bash $T/probe_cell.sh
NAME=d4-live8-churn STATE=live MODE=churn SECONDS_RUN=240 HOLD_GIB=0 READERS=8 RETRY_MAX=200 bash $T/probe_cell.sh
touch jobs/campaign-d.done
