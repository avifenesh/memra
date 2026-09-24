#!/usr/bin/env bash
# WP-B local 5090 queue (2026-09-24): after the DAY37 r3 chain (pid argument) exits, the DAY38 local chain, then the
# DAY39 local chain once day39-build.sh has written target/day39/SHA256SUMS. Each chain takes the rig lock per step
# itself. Reads /proc only; never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-b.log"; }
pid=${1:?chain-r3 pid}
log "queue start: waiting for pid $pid"
while [ -d "/proc/$pid" ]; do sleep 60; done
log "pid $pid gone; day 38 chain"
bash "$D/rtx5090-day38/chain.sh"; log "day 38 chain rc=$?"
until [ -s "$WT/target/day39/SHA256SUMS" ]; do
  [ -f /tmp/wpb-d39-build.out ] && command grep -q "failed" /tmp/wpb-d39-build.out && { log "day 39 builds failed; not run"; exit 1; }
  sleep 60
done
bash "$D/rtx5090-day39/chain.sh"; log "day 39 chain rc=$?"
log "queue done"
