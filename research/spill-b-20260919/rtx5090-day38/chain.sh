#!/usr/bin/env bash
# DAY38 local chain (1.3, addenda A and B): the four main boots both orders, the three fault boots (batched, non-batching,
# the non-batching red arm), the vmm interaction pair, then the reader. Binaries target/day38/{green,red}.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day38
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
log "chain start HEAD=$(git rev-parse HEAD) green=$(sha256sum target/day38/green/memra-server | cut -c1-16) red=$(sha256sum target/day38/red/memra-server | cut -c1-16)"
bash research/spill-b-20260919/day38-run.sh "$R" \
  main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off \
  fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red \
  vmm-off:vmm-off vmm-on:vmm-on
python3 research/spill-b-20260919/day38-read.py rtx5090 "$R" > "$R/read.log" 2>&1
log "chain done"
