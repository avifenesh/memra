#!/usr/bin/env bash
# Final-binary GPU campaign: draft cells, serial battery, and the pool cell in one window,
# three arms interleaved in both orders: serial (split), step-2 pooled verify (pool), this
# commit's pooled draft (draft).
set -u
W=$HOME/.cache/memra-batchspec-draft-work
H=$HOME/.cache/memra-batchspec-draft/research/batch-spec-20261101/battery
BAT=$HOME/.cache/memra-batchspec-battery
D=$HOME/.cache/memra-batchspec-bins/draft
P=$HOME/.cache/memra-batchspec-bins/pool
S=$HOME/.cache/memra-batchspec-bins/split
bash "$W/draft_cells.sh" "$D/decode-batch-bench" "$BAT/draft-cells-final"
bash "$HOME/.cache/memra-batchspec-harness/battery.sh" "$D" draft
port=8370
for run in draft:$D:a pool:$P:a serial:$S:a serial:$S:b pool:$P:b draft:$D:b draft:$D:c pool:$P:c serial:$S:c; do
    IFS=: read -r arm bin tag <<< "$run"
    port=$((port + 1))
    python3 "$H/pool_cell.py" "$bin/memra-server" "$BAT/ab-$arm-$tag" $port > "$BAT/ab-$arm-$tag.out" 2>&1
    echo "ab-$arm-$tag rc=$?"
done
echo "gpu final done"
