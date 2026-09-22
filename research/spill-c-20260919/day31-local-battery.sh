#!/usr/bin/env bash
# day31-local-battery.sh <out_root> [server_bin]: the whole-budget failure arm (MEMRA_KV_HOST_TENANT_PCT=100) on
# the local RTX 5090 Laptop GPU, default and plain, door OFF and ON, each cell through day31-cell.sh in one
# sitting. The 9B artifact. Pass/fail cells, no timing claim. Before every cell: wait, bounded (15 x 120 s),
# until the card carries no compute app and reports at least 20000 MiB free (the day-26 rule), logging each wait
# with the compute-apps snapshot; the holder is never inspected beyond nvidia-smi's own listing, never signalled.
set -uo pipefail
ROOT=${1:?out_root}; HERE=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${2:-$HERE/target/release/memra-server}
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
mkdir -p "$ROOT"; sha256sum "$BIN" > "$ROOT/binary.sha256"
TREE=$(git -C "$HERE" rev-parse HEAD)
wait_idle() {
    for attempt in $(seq 1 15); do
        apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then return 0; fi
        echo "$(date -u +%FT%TZ) wait $attempt before $1: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$ROOT/battery.log"
        sleep 120
    done
    echo "$(date -u +%FT%TZ) card not idle after 15 waits before $1; NOT RUN" | tee -a "$ROOT/battery.log"
    return 1
}
for cell in failure-default-pct100-off failure-default-pct100-on failure-plain-pct100-off failure-plain-pct100-on; do
    if ! wait_idle "$cell"; then
        nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu,power.draw --format=csv | tee -a "$ROOT/battery.log"
        echo "NOT RUN: $cell" | tee -a "$ROOT/battery.log"
        continue
    fi
    echo "$(date -u +%FT%TZ) start $cell tree=$TREE" | tee -a "$ROOT/battery.log"
    bash "$HERE/research/spill-c-20260919/day31-cell.sh" "$cell" "$M9" "$BIN" "$ROOT" 64 2>&1 | tee -a "$ROOT/battery.log"
    echo "$(date -u +%FT%TZ) done $cell rc=$(cat "$ROOT/$cell/gate.exit")" | tee -a "$ROOT/battery.log"
done
echo "LOCAL-BATTERY-DONE" | tee -a "$ROOT/battery.log"
