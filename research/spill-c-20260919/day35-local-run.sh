#!/usr/bin/env bash
# Day 35 local runner for the tenant-stall cell (the local RTX 5090 Laptop GPU, collector rig `rtx5090`, lock
# /tmp/memra-5090.lock): waits, bounded, for the card to carry no compute app and at least 20000 MiB free (the
# day-26 rule; another session's server may hold the card for long windows), then runs day35-stall-cell.sh ONCE
# through tools/tier-battery.py --external-lock (one lock hold, six boots). Bounded: 15 waits of 120 s in total
# across idle waits and lock refusals (30 minutes); a card that is still busy after that is recorded as
# NOT RUN with the card's last snapshot, and nothing on the card is inspected beyond nvidia-smi's own listing or
# signalled. No host, id or price here.
# usage: day35-local-run.sh <receipts_root> [server_bin]
set -uo pipefail
R=$1
HERE=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${2:-$HERE/target/release/memra-server}
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
cd "$HERE" || exit 1
mkdir -p "$R/stall"
cell=stall
rc=2
attempt=0
waits=0
while [ "$waits" -le 15 ]; do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$apps" ] || [ "${free_mib:-0}" -lt 20000 ]; then
        echo "$(date -u +%FT%TZ) wait $waits: card busy before $cell: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$R/stall/waits.log"
        waits=$((waits + 1))
        [ "$waits" -le 15 ] && sleep 120
        continue
    fi
    out=$R/stall/collector; [ "$attempt" -gt 0 ] && out=$R/stall/collector-retry$attempt
    python3 tools/tier-battery.py --rig rtx5090 --timeout 3600 --out "$out" --external-lock \
        --execute bash research/spill-c-20260919/day35-stall-cell.sh @COLLECTOR_LOCK_FD@ "$HERE" "$R" "$MODEL" "$BIN" \
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
