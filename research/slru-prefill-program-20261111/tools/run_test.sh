#!/usr/bin/env bash
# run_test.sh <test binary> <log>   one gpulock lease, the placement GPU gate
set -u
bin=$(realpath "$1"); log=$(realpath -m "$2")
cd "$HOME/.cache/memra-1217/wt" || exit 1
gpulock run --wait 3h --ttl 30m --for "memra #1217 placement gpu test" -- \
  env NVIDIA_TF32_OVERRIDE=0 "$bin" --ignored --test-threads=1 --nocapture > "$log" 2>&1
echo "RC=$?" >> "$log"
