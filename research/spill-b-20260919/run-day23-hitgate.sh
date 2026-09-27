#!/usr/bin/env bash
# Day 23 copy of the day-22/21 runner for the MERGED tree (origin/main 653c997f4, #614 small_m_tier_max; the lane's prefill_rows scope removed): same cell, same arms, receipts under rtx5090-day23/.
# Day 22 memra#427 cell F4: the #379 hit gate (tools/spec-on-cache-hit-gate.sh qwen) on the FIX memra-server, local
# RTX 5090, as on day 19. The gate takes the canonical lock itself (`flock -w 300 /tmp/memra-5090.lock` around every
# boot, as local-ci runs it), so it does not go through the collector (which would hold the same lock). The 9B trunk
# the battery uses; the external drafter local-ci attaches is absent on this rig (recorded, not hidden). Expected:
# `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (day 19's corrected run read ALL GREEN on the pre-fix tree).
# usage: run-day22-hitgate.sh <cell-name>
set -uo pipefail
cell=${1:?cell}
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day23
BIN=${BIN:-$WT/target/release/memra-server}
HITQ=${HITQ:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
HITQD=${HITQD:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
EV=$R/$cell/qwen; rm -rf "$EV"; mkdir -p "$EV"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
sha256sum "$BIN" > $R/$cell/binary.sha256; git rev-parse HEAD > $R/$cell/gate-source.txt
if [ -f "$HITQD" ]; then export MEMRA_GATE_MTP_DRAFT="$HITQD"; echo "drafter: $HITQD" > $R/$cell/drafter.txt; else echo "drafter absent: $HITQD (assertion no-ops, as local-ci's WARNING branch)" > $R/$cell/drafter.txt; fi
date -u +%FT%TZ > $R/$cell/started.txt
MEMRA_GPU_LOCK=/tmp/memra-5090.lock run bash tools/spec-on-cache-hit-gate.sh qwen "$HITQ" "$BIN" "$EV" > $R/$cell/gate.log 2>&1
rc=$?; echo $rc > $R/$cell.exit; date -u +%FT%TZ > $R/$cell/finished.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
tail -3 $R/$cell/gate.log
exit $rc
