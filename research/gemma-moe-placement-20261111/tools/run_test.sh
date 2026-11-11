#!/usr/bin/env bash
# run_test.sh <arm> <log name> [ENV=V ...] : the gemma placement GPU test under one lease.
set -u
D=$HOME/.cache/memra-1225
arm=$1; name=$2; shift 2
gpulock run --wait 3h --ttl 40m --for "memra #1225 placement test $name" -- \
  env NVIDIA_TF32_OVERRIDE=0 "$@" "$D/bin/moe_slru_gemma_placement_gpu-$arm" --ignored --test-threads=1 --nocapture \
  > "$D/gpu-test/$name.log" 2>&1
echo "rc=$?" >> "$D/gpu-test/$name.log"
