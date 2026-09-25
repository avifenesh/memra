#!/usr/bin/env bash
# DAY42 local chain (1.4, addenda A to F; the worker-level demote queue): RTX 5090, the 9B at MEMRA_CTX=65536, the host tier at 8,192 MB, burst 32 at
# max_ctx 34,816 and max_tokens 64, warm 16 x 4,096 tokens at max_tokens 16; the
# eight boots on target/day42e/tip (build-arms.sh), then the reader. Every GPU step waits for an idle rig and holds
# /tmp/memra-5090.lock alone. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day42e
cd "$WT" || exit 1
cp target/day42e/SHA256SUMS "$R/binaries.sha256"; cp target/day42e/tip/source.commit "$R/tip.source"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")" >> "$R/chain.log"
BIN=$WT/target/day42e/tip/memra-server HOST_MB=8192 BURST=32 WARM=16 WARM_TOKENS=4096 WARM_MAX_TOKENS=16 BURST_MAX_CTX=34816 \
  bash "$D/day42-run.sh" "$R" ontick-O1:ontick offtick-O1:offtick \
  offtick-O2:offtick ontick-O2:ontick ontick-nocontracts:ontick-nocontracts fault-d2h-delay:fault-d2h-delay \
  fault-d2h-source-flip:fault-d2h-source-flip fault-sources-helper-gone:fault-sources-helper-gone
python3 "$D/day42-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"
