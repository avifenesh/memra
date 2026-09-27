#!/usr/bin/env bash
# Day 29 fault arm of the twin gate's V3-premise refusal (DAY29.md 6): under ONE collector lock hold on the canonical
# /tmp/memra-5090.lock, start a self-owned co-tenant holding about the day-18 footprint, then run the 27B twin gate
# (door OFF) with --external-lock; the co-tenant is this cell's own child and is stopped by this cell. Expected,
# pre-registered: `REFUSED: V3 premise: ...` (exit 2) with turn 2 among the unequal windows.
# usage: tools/tier-battery.py --rig rtx5090 --timeout 1800 --out <dir> --external-lock --execute bash cotenant-cell.sh @COLLECTOR_LOCK_FD@ <mib>
set -uo pipefail
fd=$1; MIB=${2:-1390}
WT=${WT:-/home/avifenesh/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day29/fault-cotenant}
MODEL27=${MODEL27:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=${BIN:-$WT/target/release/memra-server}
cd "$WT"; mkdir -p "$R"
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$R/LOCK.json"
snap() { { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$R/card.txt"; }
snap before
"${COTENANT:-/tmp/spillb29/cotenant}" "$MIB" > "$R/cotenant.log" 2>&1 &
CT=$!
for i in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$CT" && break; sleep 1; done
snap cotenant-up
rm -rf "$R/twin27-off"
python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL27" --bin "$BIN" --out "$R/twin27-off" > "$R/twin27-off.log" 2>&1
rc=$?; echo "$rc" > "$R/twin27-off.exit"
snap gate-done
kill "$CT" 2>/dev/null; wait "$CT" 2>/dev/null
snap after
grep -h 'REFUSED\|PREFIX-NEWEST' "$R/twin27-off.log" | tail -1
echo "cotenant-cell rc=$rc"
