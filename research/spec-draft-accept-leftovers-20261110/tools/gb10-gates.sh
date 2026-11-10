#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the seeded-invariance gate on the after server,
# then on the before server. Exit 0 when both pass.
set -u
D=$HOME/.cache/memra-1160
M=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
bad=0
for arm in after before; do
  python3 "$D/seeded-invariance-gate.py" "$D/bin/memra-server-$arm" "$M" --out "$D/run/gate-$arm" --penalised host > "$D/run/gate-$arm.log" 2>&1 || bad=1
  tail -n 1 "$D/run/gate-$arm.log"
done
exit $bad
