#!/usr/bin/env bash
# Correctness battery for the verify-graph door, run ON the affected model (35B-A3B).
# Both arms: default (flag off) proves the wiring changed nothing, forced ON proves the
# captured trunk is self-consistent at every K.
set -uo pipefail
M=/data/memra/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
SRC=/data/memra/memra-src
OUT=/root/vgraph-gates.txt
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo "gpu lock held"; exit 1; }
cd "$SRC"
: > $OUT

echo "=== kernel-check (config pins) ===" >> $OUT
./target/release/kernel-check "$M" >> $OUT 2>&1 || echo "kernel_check exit=$?" >> $OUT
grep -acE "FAIL|MISMATCH" $OUT >> /dev/null

for arm in off on; do
  extra=""
  [ "$arm" = on ] && extra="MEMRA_SPEC_VERIFY_GRAPH=1"
  echo "=== run-spec K=1..8 self-consistency (flag=$arm) ===" >> $OUT
  for k in 1 2 3 4 5 6 7 8; do
    line=$(env $extra MEMRA_SPEC_K=$k ./target/release/run-spec "$M" 2>&1 | tail -3 | tr '\n' ' ')
    echo "K=$k [$arm] $line" >> $OUT
  done
  echo "=== run-gen argmax (flag=$arm) ===" >> $OUT
  env $extra ./target/release/run-gen "$M" 2>&1 | tail -4 >> $OUT
done
echo GATES-DONE >> $OUT
cat $OUT
