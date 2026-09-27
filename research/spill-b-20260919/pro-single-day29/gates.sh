#!/usr/bin/env bash
# WP-B day 29 target-card twin cells (one RTX PRO 6000 Blackwell), run under the collector's lock hold:
# `tools/tier-battery.py --rig pro-single --external-lock --execute bash gates.sh @COLLECTOR_LOCK_FD@`
# (A's day-18 gates.sh shape, the two 27B twin cells only). The gate inherits the FD (--external-lock); the
# patched gate samples the card itself at each boot. Door OFF = env unset; ON = MEMRA_KV_HOST_CONTRACTS=1.
set -uo pipefail
fd=$1
R=${R:-/root/spill-receipts/b-day29}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
mkdir -p "$R/gates"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/gates/LOCK.json"
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
sha256sum "$BIN" | tee "$R/gates/binary.sha256"
sha256sum tools/prefix-newest-turn-fits-gate.py | tee "$R/gates/gate.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
run() { local name=$1; shift; "$@" > "$R/gates/$name.log" 2>&1; local rc=$?; echo "$rc" > "$R/gates/$name.exit"; echo "$(date -u +%FT%TZ) $name rc=$rc"; }
run twin27-off           python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$R/gates/twin27-off"
run twin27-on            env MEMRA_KV_HOST_CONTRACTS=1 python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$R/gates/twin27-on"
echo gates-done
