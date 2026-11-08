#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one cold served 32k prefill under nsys per arm
# (v3 off, v3 on, v3 off again), then the F8F4 GEMM kernel time by grid from each trace.
# usage: prof-v3.sh BIN OUT
set -u
BIN=$1; OUT=$2
D=$HOME/.cache/memra-f8f4v3
export TMPDIR=$D/tmp PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
i=0
for arm in off on off; do
  i=$((i+1))
  extra=""; [ $arm = off ] && extra="MEMRA_MMQ_F8F4_V3=0"
  o=$OUT/$i-$arm
  bash "$D/h/profile_arm.sh" "$o" "$BIN" "$extra" 32768 > "$OUT/$i-$arm.out" 2>&1
  python3 "$D/kshapes.py" "$o/prof.sqlite" > "$o/kshapes.txt" 2>&1
  echo "== $i $arm"; cat "$o/kshapes.txt"; grep -h '^{' "$o/run.log" | cut -c1-120
  rm -f "$o/prof.sqlite"
done
echo PROF-V3-DONE
