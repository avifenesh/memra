#!/usr/bin/env bash
# Day 39 target-card runner (one RTX PRO 6000 Blackwell, collector rig `pro-single`, lock /tmp/memra-gpu.lock):
# checks the 27B artifact against its recorded SHA-256 (read only), requires the three binaries' build receipt, then
# waits, bounded, for the card to carry no compute app, and runs day39-stall-cell.sh ONCE through
# tools/tier-battery.py --external-lock (one lock hold, six programs of six boots). The collector's timeout is
# 5400 s, so the one hold cannot pass 90 minutes (DAY39.md section 1). Bounded: 15 waits of 120 s in total across
# idle waits and lock refusals (30 minutes); a card still busy after that is recorded as NOT RUN with the card's
# last snapshot, and nothing on the card is inspected beyond nvidia-smi's own listing or signalled (day 37's
# runner, the rig changed). No host, id or price here.
# usage: day39-box-run.sh <tree> <receipts_root> <model.gguf> <bin_dir>
set -uo pipefail
TREE=$1; R=$2; MODEL=$3; BINDIR=$4
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
mkdir -p "$R/stall"
{
    echo "tree=$(git rev-parse HEAD) start=$(date -u +%FT%TZ) host_utc_offset=$(date +%z)"
    want=$(cut -d' ' -f1 "$MODEL.sha256"); got=$(sha256sum "$MODEL" | cut -d' ' -f1)
    echo "model=$(basename "$MODEL") want=$want got=$got"
    [ -n "$want" ] && [ "$want" = "$got" ] && echo "model sha256 MATCH" || echo "model sha256 MISMATCH"
} | tee "$R/stall/provenance.log"
grep -q '^model sha256 MATCH$' "$R/stall/provenance.log" || { echo "NOT RUN: artifact hash mismatch" | tee "$R/stall/NOT-RUN.txt"; exit 3; }
grep -q '^BUILDS-DONE ' "$BINDIR/builds.log" 2>/dev/null || { echo "NOT RUN: builds not done" | tee "$R/stall/NOT-RUN.txt"; exit 3; }
cp "$BINDIR/builds.log" "$R/stall/builds.log"
cell=stall
rc=2
attempt=0
waits=0
while [ "$waits" -le 15 ]; do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    if [ -n "$apps" ]; then
        echo "$(date -u +%FT%TZ) wait $waits: card busy before $cell: apps=[${apps//$'\n'/; }]" | tee -a "$R/stall/waits.log"
        waits=$((waits + 1))
        [ "$waits" -le 15 ] && sleep 120
        continue
    fi
    out=$R/stall/collector; [ "$attempt" -gt 0 ] && out=$R/stall/collector-retry$attempt
    python3 tools/tier-battery.py --rig pro-single --timeout 5400 --out "$out" --external-lock \
        --execute bash research/spill-c-20260919/day39-stall-cell.sh @COLLECTOR_LOCK_FD@ "$TREE" "$R" "$MODEL" "$BINDIR" \
        > "$out-driver.log" 2>&1
    rc=$?
    echo "$rc" > "$out.exit"
    if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
        echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy" | tee -a "$R/stall/waits.log"
        rm -f "$out.exit"; rmdir "$out" 2>/dev/null
        attempt=$((attempt + 1)); waits=$((waits + 1))
        [ "$waits" -le 15 ] && sleep 120
        continue
    fi
    break
done
if [ "$waits" -gt 15 ]; then
    {
        echo "NOT RUN: the bounded wait (15 x 120 s) ended with the card busy or the lock held; last snapshot:"
        nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu,power.draw --format=csv
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    } | tee "$R/stall/NOT-RUN.txt"
    exit 3
fi
echo "$(date -u +%FT%TZ) $cell rc=$rc" | tee -a "$R/stall/progress.log"
exit "$rc"
