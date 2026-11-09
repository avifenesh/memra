#!/usr/bin/env bash
# One scored phase: interleaved fresh boots, one gpulock lease per arm (so other lanes' records
# interleave between arms).
#   A=BIN_A B=BIN_B phase.sh PHASE "A B B A A B" WORKLOAD [VAR=VALUE ...]
# WORKLOAD as arm2.sh takes it (mixed | agentic:U[:TEMP[:MAXTOK[:prime]]]). A YIELD file in the lane
# dir stops the phase at the next arm boundary.
set -u
PHASE=$1; ORDER=$2; WL=$3; shift 3
H=<harness>
O=<runs>/$PHASE
mkdir -p "$O"
n=${N0:-0}  # N0: resume a phase after its first N0 arms
for arm in $ORDER; do
  n=$((n + 1))
  if [ -e <lane>/YIELD ]; then echo "yield before $n-$arm"; break; fi
  case $arm in A) bin=$A ;; B) bin=$B ;; C) bin=${C:?} ;; *) bin=${!arm:?} ;; esac
  label=$(printf %02d-%s "$n" "$arm")
  [ -x "$bin" ] || { echo "no binary for $arm: $bin"; exit 3; }
  echo "$(TZ=Asia/Jerusalem date -Is) $PHASE $label waiting for the 5090"
  gpulock run --ttl 40m --wait 4h --for "memra c8r2 $PHASE $label (one arm, fresh boot)" -- \
    bash "$H/arm2.sh" "$label" "$bin" "$O" "$WL" "$@"
  echo "$(TZ=Asia/Jerusalem date -Is) $PHASE $label rc=$?"
done
echo "PHASE_DONE $PHASE"
