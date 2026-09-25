#!/usr/bin/env bash
# DAY42 local chain (1.4, addendum A): RTX 5090, the 9B at MEMRA_CTX=65536, the host tier at 8,192 MB, burst 32; the
# eight boots on target/day42/tip (build-arms.sh), then the reader. Every GPU step waits for an idle rig and holds
# /tmp/memra-5090.lock alone. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day42
cd "$WT" || exit 1
cp target/day42/SHA256SUMS "$R/binaries.sha256"; cp target/day42/tip/source.commit "$R/tip.source"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
BIN=$WT/target/day42/tip/memra-server HOST_MB=8192 BURST=32 bash "$D/day42-run.sh" "$R" ontick-O1:ontick offtick-O1:offtick \
  offtick-O2:offtick ontick-O2:ontick ontick-nocontracts:ontick-nocontracts fault-d2h-delay:fault-d2h-delay \
  fault-d2h-source-flip:fault-d2h-source-flip fault-sources-helper-gone:fault-sources-helper-gone
python3 "$D/day42-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
