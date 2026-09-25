#!/usr/bin/env bash
# DAY47 section 3a: the pause gate's re-run (the gate revised: await_line on a fixed string; the plain race asserts
# the program the first run read), both boots, on the same v binary, in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash pause-gate-rerun.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-v
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/v/memra-server
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
sha256sum "$BIN" tools/kv-host-pause-demote-gate.sh > "$R/gates/pause-demote-rerun.sha256"
env MEMRA_HOSTGATE_CACHE_MB=256 tools/kv-host-pause-demote-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/pause-demote-rerun" > "$R/gates/pause-demote-rerun.log" 2>&1
rc=$?; echo "$rc" > "$R/gates/pause-demote-rerun.exit"; echo "pause-demote-rerun rc=$rc"
