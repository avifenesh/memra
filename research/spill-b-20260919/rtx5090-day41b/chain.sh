#!/usr/bin/env bash
# DAY41 addenda B and C local chain (DAY41.md 1.8 and 1.9): RTX 5090, the 9B at MEMRA_CTX=65536 (day41-run.sh's local
# defaults), the RX boots per route in both orders and offprev on target/day41b (build-arms.sh: the revised tip and the
# tip plus day41b-nodoor.patch), then the reader. Every GPU step holds /tmp/memra-5090.lock alone. Never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day41b
cd "$WT" || exit 1
cp target/day41b/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
for route in plain spec; do
  BIN=$WT/target/day41b/tip/memra-server PREV_BIN=$WT/target/day41b/offprev/memra-server bash "$D/day41-run.sh" "$R" \
    "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" "rx-$route-O2-rewind:rewind:$route:RX" \
    "rx-$route-O2-keep:keep:$route:RX"
  echo "$(date -u +%FT%TZ) $route boots rc=$?" >> "$R/chain.log"
done
BIN=$WT/target/day41b/tip/memra-server PREV_BIN=$WT/target/day41b/offprev/memra-server bash "$D/day41-run.sh" "$R" offprev:offprev:plain:RX6
python3 "$D/day41-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
