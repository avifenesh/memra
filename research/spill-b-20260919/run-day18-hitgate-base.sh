#!/usr/bin/env bash
# Day 18: the #379 gate (tools/spec-on-cache-hit-gate.sh qwen) on the fix binary, local RTX 5090. The gate
# takes the canonical lock itself (`flock -w 300 /tmp/memra-5090.lock` around every boot, as local-ci runs
# it), so it does not go through the collector (which would hold the same lock). The 9B trunk the battery
# uses; the external drafter local-ci attaches is absent on this rig today, so the drafter-attach assertion
# no-ops exactly as local-ci's WARNING branch says (recorded, not hidden). Evidence under the receipt dir.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day18
BIN=${BIN:-$WT/target/bins/base/memra-server}
HITQ=${HITQ:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
HITQD=${HITQD:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
EV=$R/hitgate-base/qwen; rm -rf "$EV"; mkdir -p "$EV"
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-hitgate.csv
sha256sum "$BIN" > $R/hitgate-base/binary.sha256
if [ -f "$HITQD" ]; then export MEMRA_GATE_MTP_DRAFT="$HITQD"; echo "drafter: $HITQD" > $R/hitgate-base/drafter.txt; else echo "drafter absent: $HITQD (assertion no-ops, as local-ci's WARNING branch)" > $R/hitgate-base/drafter.txt; fi
MEMRA_GPU_LOCK=/tmp/memra-5090.lock run bash tools/spec-on-cache-hit-gate.sh qwen "$HITQ" "$BIN" "$EV" > $R/hitgate-base/gate.log 2>&1
rc=$?; echo $rc > $R/hitgate-base.exit; echo "hitgate-base rc=$rc" >> $R/chain.log
tail -3 $R/hitgate-base/gate.log
