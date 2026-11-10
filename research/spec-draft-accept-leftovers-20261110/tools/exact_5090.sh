#!/usr/bin/env bash
# Runs under the caller's GPU lease on the RTX 5090: one exactness part of memra#1169.
#   gates   the seeded-invariance gate on the after server, then on the before server
#   eager   the same gate with the eager draft chain forced (MEMRA_SPEC_NOGRAPH=1), after then
#           before, then the pooled-draft probe on the after build
# Each gate writes its own log and exits 0 on PASS; this script runs every part and fails if any
# part failed. usage: exact_5090.sh gates|eager
set -u
H=$HOME/.cache/memra-1160/run
O=$H/rtx5090
B=$HOME/.cache/memra-1160/bin
M=/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
GATE=$H/seeded-invariance-gate.py
mkdir -p "$O"
bad=0
case $1 in
  gates)
    for arm in after before; do
      sha256sum "$B/memra-server-$arm" > "$O/gate-$arm.bin.sha256"
      python3 "$GATE" "$B/memra-server-$arm" "$M" --out "$O/gate-$arm" > "$O/gate-$arm.log" 2>&1 || bad=1
      tail -n 1 "$O/gate-$arm.log"
    done ;;
  eager)
    for arm in after before; do
      sha256sum "$B/memra-server-$arm" > "$O/gate-eager-$arm.bin.sha256"
      python3 "$GATE" "$B/memra-server-$arm" "$M" --out "$O/gate-eager-$arm" --env MEMRA_SPEC_NOGRAPH=1 > "$O/gate-eager-$arm.log" 2>&1 || bad=1
      tail -n 1 "$O/gate-eager-$arm.log"
    done
    bash "$H/probe_cells.sh" "$B/decode-batch-bench-after" "$M" "$O/probe" > "$O/probe.log" 2>&1 || bad=1
    tail -n 1 "$O/probe.log" ;;
esac
exit $bad
