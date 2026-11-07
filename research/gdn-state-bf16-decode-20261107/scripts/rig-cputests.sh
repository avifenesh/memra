#!/usr/bin/env bash
# Rig: the CPU unit-test suites the lane touches (engine and server lib, memra-kv), release
# profile beside the lane build, under the shared build lock, 2 jobs at nice 15.
set -u
R=$HOME/.cache/memra-gdnbf16-decode
export CARGO_TARGET_DIR=$R/target-120a
L=$R/logs/cputests.log
cd "$R/wt" || exit 1
{
  echo "start $(date '+%F %T') HEAD=$(git rev-parse HEAD)"
  for p in memra-kv memra-engine memra-server; do
    flock ~/.cache/memra-rig-build.lock nice -n 15 cargo test --release --locked -j 2 -p $p --lib > "$R/logs/cputest-$p.log" 2>&1
    echo "$p rc=$? $(grep -E '^test result' "$R/logs/cputest-$p.log" | tail -n 1)"
    grep -E '^test .* FAILED$|panicked at' "$R/logs/cputest-$p.log" | head -n 10
  done
  echo CPUTESTS_DONE
} > "$L" 2>&1
