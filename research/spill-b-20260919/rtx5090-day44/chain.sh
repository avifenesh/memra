#!/usr/bin/env bash
# DAY44 local chain (DAY44.md 1.6): RTX 5090, the 9B at MEMRA_CTX=65536 (day44-run.sh's local defaults), lengths 6,144 and
# 30,720; per route and shape the keep and exact boots in both orders, offprev and the settle fault boot, on target/day44
# (build-arms.sh: the tip and the tip plus day44-nodoor.patch); then the reader. Every GPU step holds
# /tmp/memra-5090.lock alone. Never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day44
cd "$WT" || exit 1
cp target/day44/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
export BIN=$WT/target/day44/tip/memra-server PREV_BIN=$WT/target/day44/offprev/memra-server
for route in plain spec; do
  for shape in rx rxg; do
    S=$( [ "$shape" = rx ] && echo RX || echo RXg )
    bash "$D/day44-run.sh" "$R" "rx-$route-$shape-O1-keep:keep:$route:$S" "rx-$route-$shape-O1-exact:exact:$route:$S" \
      "rx-$route-$shape-O2-exact:exact:$route:$S" "rx-$route-$shape-O2-keep:keep:$route:$S"
    echo "$(date -u +%FT%TZ) $route $shape boots rc=$?" >> "$R/chain.log"
  done
done
bash "$D/day44-run.sh" "$R" offprev:offprev:plain:RX6 fault-plain-rxg6:fault:plain:RXg6
python3 "$D/day44-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
