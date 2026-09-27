#!/usr/bin/env bash
# WP-B local 5090 queue-f (2026-09-25): after queue-e (pid argument) has exited, the DAY42 local chain (queue-e itself
# waits for the card's health and runs every earlier cell). Reads /proc only; never a signal.
set -uo pipefail
D=$HOME/projects/wt-spill-b/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-f.log"; }
pid=${1:?queue-e pid}
log "queue-f start: waiting for queue-e pid $pid"
while [ -d "/proc/$pid" ]; do sleep 120; done
until [ -s "$HOME/projects/wt-spill-b/target/day42/SHA256SUMS" ]; do sleep 120; done
bash "$D/rtx5090-day42/chain.sh"; log "day 42 chain rc=$?"
log "queue-f done"
