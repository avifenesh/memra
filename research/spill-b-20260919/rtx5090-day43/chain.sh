#!/usr/bin/env bash
# DAY43 local chain (DAY43.md 1.4, O13): RTX 5090, the 9B at MEMRA_CTX=65536 (day43-run.sh's local defaults), the
# spec-route RX boots unset and clamp in both orders and offprev on target/day43 (build-arms.sh: the tip and the tip plus
# day43-nodoor.patch), then the reader. Every GPU step holds /tmp/memra-5090.lock alone. Never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day43
cd "$WT" || exit 1
cp target/day43/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
BIN=$WT/target/day43/tip/memra-server PREV_BIN=$WT/target/day43/offprev/memra-server bash "$D/day43-run.sh" "$R" \
  rx-spec-O1-unset:unset:spec:RX rx-spec-O1-clamp:clamp:spec:RX rx-spec-O2-clamp:clamp:spec:RX \
  rx-spec-O2-unset:unset:spec:RX offprev:offprev:spec:RX
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$R/chain.log"
python3 "$D/day43-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
