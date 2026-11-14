#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign C, the serving cycles both
# occurrences came from, arms alternating: T (the head: context creation retried on unified
# memory) and B (the red arm: the same tree with the retry budget at zero).
# usage: campaign_c.sh NAME CYCLES   (from ~/.cache/memra-1316; ends by creating jobs/NAME.done)
set -u
T=research/gb10-boot-oom-20261114/tools
NAME=$1 CYCLES=$2 ARMS="T B" bash $T/serve_cycle.sh
touch "jobs/$1.done"
