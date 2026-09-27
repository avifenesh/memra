#!/usr/bin/env bash
# WP-B local 5090 queue (2026-09-24, addendum E): once build-r4.sh has written every day's binaries (r4-binaries.source
# and target/day39/SHA256SUMS), the DAY37 r4 chain, then the DAY38 local chain, then the DAY39 local chain, in the order
# of their decide-by dates. Each chain takes the rig lock per step itself. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-b.log"; }
log "queue start (r4): waiting for the r4 builds"
until [ -s "$D/rtx5090-day37/r4-binaries.source" ] && [ -s "$WT/target/day39/SHA256SUMS" ]; do
  command grep -q "failed" /tmp/wpb-build-r4.out 2>/dev/null && { log "r4 builds failed: $(tail -1 /tmp/wpb-build-r4.out)"; exit 1; }
  sleep 60
done
log "r4 builds present: $(tr '\n' ' ' < "$D/rtx5090-day37/r4-binaries.sha256" | cut -c1-200)"
bash "$D/rtx5090-day37/chain-r4.sh"; log "day 37 r4 chain rc=$?"
bash "$D/rtx5090-day38/chain.sh"; log "day 38 chain rc=$?"
bash "$D/rtx5090-day39/chain.sh"; log "day 39 chain rc=$?"
log "queue done"
