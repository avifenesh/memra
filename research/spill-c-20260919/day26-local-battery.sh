#!/usr/bin/env bash
# day26-local-battery.sh <out_root> [server_bin]: the door gates lane A's day 20 owed on the local RTX 5090
# Laptop GPU, on the Move 2 slice-2 tree (A `ff3f0f6f5` through integ37), each cell through day26-cell.sh in one
# sitting, plus the restore route census per cell. The 9B artifact for the host-tier gates and the hit gate; the
# 27B for the twin gate (the 9B twin refuses `cohort promotion did not happen`). Pass/fail cells, no timing claim.
set -uo pipefail
ROOT=${1:?out_root}; HERE=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${2:-$HERE/target/release/memra-server}
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$ROOT"; sha256sum "$BIN" > "$ROOT/binary.sha256"
TREE=$(git -C "$HERE" rev-parse HEAD)
# Attempt 1 today booted twelve cells into CUDA_ERROR_OUT_OF_MEMORY while a foreign process held 22176 MiB of
# the card (the gate's flock was free; the holder ran outside it). So before every cell: wait, bounded
# (15 x 120 s), until the card carries no compute app and reports at least 20000 MiB free, logging each wait
# with the compute-apps snapshot; the holder is never inspected beyond nvidia-smi's own listing, never signalled.
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
for cell in fault-default fault-plain \
            failure-default-off failure-default-on failure-plain-off failure-plain-on \
            identity-default-off identity-default-on identity-plain-off identity-plain-on \
            hit-off hit-on twin27-off twin27-on unit-server unit-engine; do
    wait_idle "$cell"
    echo "$(date -u +%FT%TZ) start $cell tree=$TREE" | tee -a "$ROOT/battery.log"
    bash "$HERE/research/spill-c-20260919/day26-cell.sh" "$cell" "$M9" "$M27" "$BIN" "$ROOT" 64 2>&1 | tee -a "$ROOT/battery.log"
    echo "$(date -u +%FT%TZ) done $cell rc=$(cat "$ROOT/$cell/gate.exit")" | tee -a "$ROOT/battery.log"
done
echo "LOCAL-BATTERY-DONE" | tee -a "$ROOT/battery.log"
