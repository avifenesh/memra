#!/usr/bin/env bash
# DAY45 local chain (DAY45.md 1.4): RTX 5090, the 9B at MEMRA_CTX=65536, burst 32 of 6,144 tokens, the W-release boots off
# and on in both orders on target/day45 (build-arms.sh), then the reader. Every GPU step holds /tmp/memra-5090.lock
# alone. Never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day45
cd "$WT" || exit 1
cp target/day45/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
BIN=$WT/target/day45/tip/memra-server CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64 --wave2-delay-s 10" DOOR_MEMORY=1 \
  bash "$D/day45-run.sh" "$R" O1-off:off O1-on:on O2-on:on O2-off:off
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$R/chain.log"
python3 "$D/day45-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
