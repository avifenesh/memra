#!/usr/bin/env bash
# run_kq.sh <arm> <log name> [ENV=V ...] : the k-quant gemma4_moe GPU test under one lease.
# <arm> names a built test binary: `red` is the test on the base engine, `fix` on the fix.
set -u
D=${KQ_DIR:?directory holding bin/ and kq/}
arm=$1; name=$2; shift 2
gpulock run --wait 3h --ttl 30m --for "memra #1234 k-quant test $name" -- \
  env NVIDIA_TF32_OVERRIDE=0 "$@" "$D/bin/gemma4_moe_kquant_pairs_gpu-$arm" --ignored --test-threads=1 --nocapture \
  > "$D/kq/$name.log" 2>&1
echo "rc=$?" >> "$D/kq/$name.log"
