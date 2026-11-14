#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign A, driver-level boots.
#   a0  the host as found, 200 boots (control)
#   a1  page cache full, two readers live, 400 boots 200 ms apart
#   a2  page cache full, two readers live, 15 times: a 30 GiB holder exits, then a boot every
#       second for 20 s (the "15 s after the previous server exited" timing of both occurrences)
# A failed call is retried in place up to 40 times, 50 ms apart, to measure how long the state lasts.
# usage: campaign_a.sh   (from ~/.cache/memra-1316; ends by creating jobs/campaign-a.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=a0-asis-loop STATE=asis MODE=loop N=200 GAP_MS=200 RETRY_MAX=40 bash $T/probe_cell.sh
NAME=a1-live2-loop STATE=live MODE=loop N=400 GAP_MS=200 READERS=2 RETRY_MAX=40 bash $T/probe_cell.sh
NAME=a2-live2-exit STATE=live MODE=exit TRIALS=15 HOLD_GIB=30 HOLD_S=10 BURST=20 READERS=2 RETRY_MAX=40 bash $T/probe_cell.sh
touch jobs/campaign-a.done
