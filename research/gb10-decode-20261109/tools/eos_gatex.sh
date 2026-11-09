#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the session gate's turn-2 mismatch, located.
# Both lever 1 builds (B = main, T = the lever; the engine is the same) at K=3 / 48 new tokens
# and at the gate's own defaults (K=2, 24). usage: eos_gatex.sh CAMPAIGN BASE10 HEAD10
set -u
D=$HOME/.cache/memra-gb10-r4
CAMP=$1; B=$D/bins/eosb-$2; T=$D/bins/eost-$3
CD=$D/runs/$CAMP; mkdir -p "$CD"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for arm in B T; do
  bin=$B; [ "$arm" = T ] && bin=$T
  (MEMRA_SPEC_K=3 MEMRA_NGEN=48 "$bin/session-gate" "$G8") > "$CD/session-gate-k3-$arm.log" 2>&1
  echo "session-gate k3/48 $arm rc=$? $(tail -n 1 "$CD/session-gate-k3-$arm.log")" >> "$CD/campaign.log"
  ("$bin/session-gate" "$G8") > "$CD/session-gate-def-$arm.log" 2>&1
  echo "session-gate default $arm rc=$? $(tail -n 1 "$CD/session-gate-def-$arm.log")" >> "$CD/campaign.log"
done
echo "EOS-CELL-DONE gatex" >> "$CD/campaign.log"
