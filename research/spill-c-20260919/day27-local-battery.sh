#!/usr/bin/env bash
# day27-local-battery.sh <out_root> [server_bin]: the hit gate OFF and ON on the local RTX 5090 Laptop GPU (9B)
# on the tree whose hit gate arms the host tier in its door ON arm (C day 27), each cell through day27-cell.sh.
# Before every cell: wait, bounded (15 x 120 s), until the card carries no compute app and reports at least
# 20000 MiB free (the day-26 rule, after a foreign 22 GB holder outside the lock booted twelve cells into
# CUDA_ERROR_OUT_OF_MEMORY); the holder is never inspected beyond nvidia-smi's own listing, never signalled.
# Pass/fail cells, no timing claim.
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
    echo "$(date -u +%FT%TZ) card not idle after 15 waits before $1; the cell runs anyway and records its own snapshots" | tee -a "$ROOT/battery.log"
}
for cell in hit-off hit-on; do
    wait_idle "$cell"
    echo "$(date -u +%FT%TZ) start $cell tree=$TREE" | tee -a "$ROOT/battery.log"
    bash "$HERE/research/spill-c-20260919/day27-cell.sh" "$cell" "$M9" "$BIN" "$ROOT" 2>&1 | tee -a "$ROOT/battery.log"
    echo "$(date -u +%FT%TZ) done $cell rc=$(cat "$ROOT/$cell/gate.exit")" | tee -a "$ROOT/battery.log"
done
echo "LOCAL-BATTERY-DONE" | tee -a "$ROOT/battery.log"
