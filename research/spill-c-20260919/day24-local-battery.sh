#!/usr/bin/env bash
# day24-local-battery.sh <out_root> [server_bin]: the door gates lane A's day 20 owed on the local RTX 5090
# Laptop GPU, on the slice-1 tree (A `62aa92279` merged), each cell through day24-cell.sh in one sitting.
# The 9B artifact for the host-tier gates and the hit gate; the 27B for the twin gate (the 9B twin refuses
# `cohort promotion did not happen`, A's day-17 shape fact). Pass/fail cells, no timing claim.
set -uo pipefail
ROOT=${1:?out_root}; HERE=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${2:-$HERE/target/release/memra-server}
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$ROOT"; sha256sum "$BIN" > "$ROOT/binary.sha256"
TREE=$(git -C "$HERE" rev-parse HEAD)
for cell in fault-default fault-plain \
            failure-default-off failure-default-on failure-plain-off failure-plain-on \
            identity-default-off identity-default-on identity-plain-off identity-plain-on \
            hit-off hit-on twin27-off twin27-on unit-server unit-engine; do
    echo "$(date -u +%FT%TZ) start $cell tree=$TREE" | tee -a "$ROOT/battery.log"
    bash "$HERE/research/spill-c-20260919/day24-cell.sh" "$cell" "$M9" "$M27" "$BIN" "$ROOT" 64 2>&1 | tee -a "$ROOT/battery.log"
    echo "$(date -u +%FT%TZ) done $cell rc=$(cat "$ROOT/$cell/gate.exit")" | tee -a "$ROOT/battery.log"
done
echo "LOCAL-BATTERY-DONE" | tee -a "$ROOT/battery.log"
